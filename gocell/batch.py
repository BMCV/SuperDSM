import gocell.config   as config
import gocell.pipeline as pipeline
import gocell.aux      as aux
import gocell.io       as io
import gocell.render   as render
import sys, os, pathlib, json, gzip, dill, tempfile, subprocess, skimage, warnings, csv, hashlib
import ray
import numpy as np


def load_xcf_layer(xcf_path, layername):
    with tempfile.NamedTemporaryFile() as png_file:
        subprocess.call(['xcf2png', xcf_path, layername, '-o', png_file.name])
        img = skimage.io.imread(png_file.name, plugin='matplotlib', as_grey=True, format='png')
        if img is None: warnings.warn('couldn\'t load XCF layer "%s" from file: %s' % (layername, xcf_path))
        return img

def load_unlabeled_xcf_gt(xcf_path, layername='foreground'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        foreground = load_xcf_layer(xcf_path, layername)
    regions = ndi.label(foreground, structure=skimage.morphology.disk(1))[0]
    assert all((regions == 0) == (foreground == 0))
    return regions


def load_gt(loader, filepath, **loader_kwargs):
    if loader == 'default':
        return io.imread(filepath)
    elif loader == 'xcf':
        return load_unlabeled_xcf_gt(filepath, **loader_kwargs)


@ray.remote
def _evaluate_chunk(study, chunk_id, g, candidates, gt_pathpattern, gt_is_unique, gt_loader, gt_loader_kwargs, rasterize_kwargs):
    data_chunk = dict(g=g, postprocessed_candidates=candidates)
    actual = render.rasterize_labels(data_chunk, **rasterize_kwargs)
    expected = load_gt(gt_loader, filepath=gt_pathpattern % chunk_id, **gt_loader_kwargs)
    study.set_expected(expected, unique=gt_is_unique)
    study.process(actual, unique=True, chunk_id=chunk_id)
    return chunk_id, study


def evaluate(data, gt_pathpattern, gt_is_unique, gt_loader, gt_loader_kwargs, rasterize_kwargs, out=None):
    out = ConsoleOutput.get(out)
    import segmetrics

    segmetrics.detection.FalseMerge.ACCUMULATIVE    = False
    segmetrics.detection.FalseSplit.ACCUMULATIVE    = False
    segmetrics.detection.FalsePositive.ACCUMULATIVE = False
    segmetrics.detection.FalseNegative.ACCUMULATIVE = False

    study = segmetrics.study.Study()
    study.add_measure(segmetrics.regional.Dice()        , 'Dice')
    study.add_measure(segmetrics.regional.ISBIScore()   , 'SEG')
    study.add_measure(segmetrics.regional.RandIndex()   , 'Rand')
    study.add_measure(segmetrics.regional.JaccardIndex(), 'Jaccard')
    study.add_measure(segmetrics.boundary.ObjectBasedDistance(segmetrics.boundary.NSD()           ), 'NSD')
    study.add_measure(segmetrics.boundary.ObjectBasedDistance(segmetrics.boundary.Hausdorff('a2e')), 'HSD (a2e)')
    study.add_measure(segmetrics.boundary.ObjectBasedDistance(segmetrics.boundary.Hausdorff('e2a')), 'HSD (e2a)')
    study.add_measure(segmetrics.detection.FalseSplit()   , 'd/Split')
    study.add_measure(segmetrics.detection.FalseMerge()   , 'd/Merge')
    study.add_measure(segmetrics.detection.FalsePositive(), 'd/FP')
    study.add_measure(segmetrics.detection.FalseNegative(), 'd/FN')

    study_id  = ray.put(study)
    chunk_ids = sorted(data.keys())
    futures   = [_evaluate_chunk.remote(study_id, chunk_id, data[chunk_id]['g'], data[chunk_id]['postprocessed_candidates'], str(gt_pathpattern), gt_is_unique, gt_loader, gt_loader_kwargs, rasterize_kwargs) for chunk_id in chunk_ids]
    for ret_idx, ret in enumerate(aux.get_ray_1by1(futures)):
        study_chunk = ret[1]
        study.merge(study_chunk, chunk_ids=[ret[0]])
        out.intermediate(f'Evaluated {ret_idx + 1} / {len(futures)}')
    return study


def _process_file(dry, *args, out=None, **kwargs):
    if dry:
        out = ConsoleOutput.get(out)
        out.write(f'{_process_file.__name__}: {json.dumps(kwargs)}')
    else:
        return __process_file(*args, **kwargs)


def __process_file(pipeline, data, im_filepath, seg_filepath, seg_border, log_filepath, config, first_stage, out=None):
    aux.mkdir(pathlib.Path(seg_filepath).parents[0])
    aux.mkdir(pathlib.Path(log_filepath).parents[0])
    g_raw = io.imread(im_filepath)
    out = ConsoleOutput.get(out)
    result_data = pipeline.process_image(g_raw, data=data, cfg=config, first_stage=first_stage, log_root_dir=log_filepath, out=out)[0]
    if seg_filepath is not None:
        if seg_border is None: seg_border = 8
        im_result = render.render_model_shapes_over_image(result_data, border=seg_border)
        aux.mkdir(pathlib.Path(seg_filepath).parents[0])
        io.imwrite(seg_filepath, im_result)
    return result_data


class Task:
    def __init__(self, path, data, parent_task=None):
        self.runnable    = 'runnable' in data and bool(data['runnable']) == True
        self.parent_task = parent_task
        self.path = path
        self.data = data if parent_task is None else config.derive(parent_task.data, data)
        if self.runnable:
            self.         backend = data['backend']
            self.  im_pathpattern = os.path.expanduser(data['im_pathpattern'])
            self.  gt_pathpattern = os.path.expanduser(data['gt_pathpattern'])
            self.    gt_is_unique = data['gt_is_unique']
            self.       gt_loader = data['gt_loader']
            self.gt_loader_kwargs = data['gt_loader_kwargs'] if 'gt_loader_kwargs' in data else {}
            self. seg_pathpattern = path / data['seg_pathpattern'] if 'seg_pathpattern' in data else None
            self. log_pathpattern = path / data['log_pathpattern']
            self.        file_ids = sorted(frozenset(data['file_ids']))
            self.     result_path = path / 'data.dill.gz'
            self.      study_path = path / 'study.cvs'
            self.     digest_path = path / '.digest'
            self.          config = data['config']
            self.      seg_border = data['seg_border'] if 'seg_border' in data else None
            self.          dilate = data['dilate']
            self. merge_threshold = data['merge_overlap_threshold']

    def _initialize(self):
        if self.backend == 'ray':
            ray.init(num_cpus=self.data['num_cpus'], log_to_driver=False, logging_level=ray.logging.ERROR)
            _pipeline = pipeline.create_default_pipeline('ray', selection_type='minsetcover')
            del _pipeline.stages[_pipeline.find('superpixels_entropy')]
            del _pipeline.stages[_pipeline.find('superpixels_discard')]
            return _pipeline
        return ValueError(f'unknown backend "{self.backend}"')

    def _shutdown(self):
        if self.backend == 'ray':
            ray.shutdown()
        return ValueError(f'unknown backend "{self.backend}"')

    def run(self, dry=False, verbosity=0, out=None):
        out = ConsoleOutput.get(out)
        if not self.runnable: return
        config_digest = hashlib.md5(json.dumps(self.config).encode('utf8')).hexdigest()
        if self.digest_path.exists() and self.digest_path.read_text() == config_digest:
            out.write(f'\nSkipping task: {self.path}')
            return
        out.write(f'\nEntering task: {self.path}')
        out2 = out.derive(margin=2)
        pipeline = self._initialize()
        try:
            first_stage, data = self.find_first_stage_name(pipeline, dry, out=out2)
            out3 = out2.derive(margin=2, muted = (verbosity < 0))
            for file_id in self.file_ids:
                out3.write(f'\nProcessing file ID: {file_id}')
                kwargs = dict( im_filepath = str(self. im_pathpattern) % file_id,
                              seg_filepath = str(self.seg_pathpattern) % file_id if self.seg_pathpattern is not None else None,
                              log_filepath = str(self.log_pathpattern) % file_id,
                                seg_border = self.seg_border,
                              config = config.derive(self.config, {}))
                if file_id not in data: data[file_id] = None
                data[file_id] = _process_file(dry, pipeline, data[file_id], first_stage=first_stage, out=out3, **kwargs)
            if first_stage is not None and pipeline.find(first_stage) > pipeline.find('process_candidates'):
                out2.write('\nSkipping writing results')
            else:
                out2.write(f'\nResults written to: {self.result_path}')
                if not dry:
                    with gzip.open(self.result_path, 'wb') as fout:
                        dill.dump(data, fout, byref=True)
            if not dry:
                study = evaluate(data, self.gt_pathpattern, self.gt_is_unique, self.gt_loader, self.gt_loader_kwargs, dict(merge_overlap_threshold=self.merge_threshold, dilate=self.dilate, out=out2))
                self.write_evaluation_results(data.keys(), study)
                self.digest_path.write_text(config_digest)
            out2.write(f'Evaluation study written to: {self.study_path}')
        finally:
            self._shutdown()

    def find_runnable_parent_task(self):
        if self.parent_task is None: return None
        elif self.parent_task.runnable: return self.parent_task
        else: return self.parent_task.find_runnable_parent_task()

    def find_parent_task_with_result(self):
        runnable_parent_task = self.find_runnable_parent_task()
        if runnable_parent_task is None: return None
        elif runnable_parent_task.result_path.exists(): return runnable_parent_task
        else: return runnable_parent_task.find_parent_task_with_result()

    def find_first_stage_name(self, pipeline, dry=False, out=None):
        out = ConsoleOutput.get(out)
        stage_names = [stage.name for stage in pipeline.stages]
        previous_task = self.find_parent_task_with_result()
        if previous_task is None: return None, {}
        first_stage_name = ''
        for stage_name in stage_names:
            if stage_name in self.config and (stage_name not in previous_task.config or \
                                              self.config[stage_name] != previous_task.config[stage_name]):
                first_stage_name = stage_name
                break
        data = {}
        if pipeline.find(first_stage_name) >= pipeline.find('process_candidates'):
            out.write(f'Picking up from: {previous_task.result_path} ({first_stage_name if first_stage_name != "" else "evaluate"})')
            if not dry:
                with gzip.open(previous_task.result_path, 'rb') as fin:
                    data = dill.load(fin)
        return first_stage_name, data

    def write_evaluation_results(self, chunk_ids, study):
        measure_names = sorted(study.measures.keys())
        rows = [['ID'] + measure_names]
        for chunk_id in chunk_ids:
            row = [chunk_id] + [np.mean(study.results[measure_name][chunk_id]) for measure_name in measure_names]
            rows.append(row)
        rows.append([''])
        for measure_name in measure_names:
            measure = study.measures[measure_name]
            fnc = np.sum if measure.ACCUMULATIVE else np.mean
            val = fnc(study[measure_name])
            rows[-1].append(val)
        with self.study_path.open('w', newline='') as fout:
            csv_writer = csv.writer(fout, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in rows:
                csv_writer.writerow(row)


class BatchLoader:

    def __init__(self):
        self.tasks = []

    def load(self, path):
        self.process_directory(pathlib.Path(path))

    def process_directory(self, current_dir, parent_task=None):
        task_file = current_dir / 'task.json'
        if task_file.exists():
            try:
                with task_file.open('r') as task_fin:
                    task_data = json.load(task_fin)
                task = Task(current_dir, task_data, parent_task)
            except json.JSONDecodeError as err:
                raise ValueError(f'Error processing: "{task_file}"')
            self.tasks.append(task)
            parent_task = task
        for d in os.listdir(current_dir):
            f = current_dir / d
            if f.is_dir():
                self.process_directory(f, parent_task)


class ConsoleOutput:
    def __init__(self, muted=False, parent=None, margin=0):
        self.parent = parent
        self._muted = muted
        self.margin = margin
    
    @staticmethod
    def get(out):
        return ConsoleOutput() if out is None else out

    def intermediate(self, line):
        if not self.muted: print(' ' * self.margin + line, end='\r')
    
    def write(self, line):
        if not self.muted:
            lines = line.split('\n')
            if len(lines) == 1:
                sys.stdout.write("\033[K");
                print(' ' * self.margin + line)
            else:
                for line in lines: self.write(line)

    @property
    def muted(self):
        return self._muted or (self.parent is not None and self.parent.muted)
    
    def derive(self, muted=False, margin=0):
        assert margin >= 0
        return ConsoleOutput(muted, self, self.margin + margin)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='root directory for batch processing')
    parser.add_argument('--run', help='run batch processing', action='store_true')
    parser.add_argument('--verbosity', help='postive (negative) is more (less) verbose', type=int, default=0)
    args = parser.parse_args()

    loader = BatchLoader()
    loader.load(args.path)

    dry = not args.run
    out = ConsoleOutput()
    runnable_tasks = [task for task in loader.tasks if task.runnable]
    out.write(f'Loaded {len(runnable_tasks)} runnable task(s)')
    if dry: out.write(f'DRY RUN: use "--run" to run the tasks instead')
    for task in loader.tasks:
        task.run(dry, args.verbosity, out)


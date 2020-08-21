import gocell.config   as config
import gocell.pipeline as pipeline
import gocell.aux      as aux
import gocell.io       as io
import gocell.render   as render
import sys, os, pathlib, json, gzip, dill, tempfile, subprocess, skimage, warnings, csv, hashlib
import ray
import numpy as np
import scipy.ndimage as ndi


def _format_runtime(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


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
    assert np.all((regions == 0) == (foreground == 0))
    return regions


def load_gt(loader, filepath, **loader_kwargs):
    if loader == 'default':
        return io.imread(filepath)
    elif loader == 'xcf':
        return load_unlabeled_xcf_gt(filepath, **loader_kwargs)


def evaluate(data, gt_pathpattern, gt_is_unique, gt_loader, gt_loader_kwargs, rasterize_kwargs, fast=False, out=None):
    out = aux.ConsoleOutput.get(out)
    import segmetrics

    segmetrics.detection.FalseMerge.ACCUMULATIVE    = False
    segmetrics.detection.FalseSplit.ACCUMULATIVE    = False
    segmetrics.detection.FalsePositive.ACCUMULATIVE = False
    segmetrics.detection.FalseNegative.ACCUMULATIVE = False

    study = segmetrics.study.Study()
    study.add_measure(segmetrics. regional.Dice()         , 'Dice')
    study.add_measure(segmetrics. regional.JaccardIndex() , 'Jaccard')
    if not fast:
        study.add_measure(segmetrics. regional.RandIndex()    , 'Rand')
        study.add_measure(segmetrics. regional.ISBIScore()    , 'SEG')
        study.add_measure(segmetrics.detection.FalseSplit()   , 'd/Split')
        study.add_measure(segmetrics.detection.FalseMerge()   , 'd/Merge')
        study.add_measure(segmetrics.detection.FalsePositive(), 'd/FP')
        study.add_measure(segmetrics.detection.FalseNegative(), 'd/FN')
        study.add_measure(segmetrics.boundary.ObjectBasedDistance(segmetrics.boundary.NSD()           ), 'NSD')
        study.add_measure(segmetrics.boundary.ObjectBasedDistance(segmetrics.boundary.Hausdorff('a2e')), 'HSD (a2e)')
        study.add_measure(segmetrics.boundary.ObjectBasedDistance(segmetrics.boundary.Hausdorff('e2a')), 'HSD (e2a)')

    chunk_ids = sorted(data.keys())
    for chunk_idx, chunk_id in enumerate(chunk_ids):
        actual = render.rasterize_labels(data[chunk_id], **rasterize_kwargs)
        expected = load_gt(gt_loader, filepath=gt_pathpattern % chunk_id, **gt_loader_kwargs)
        study.set_expected(expected, unique=gt_is_unique)
        study.process(actual, unique=True, chunk_id=chunk_id)
        out.intermediate(f'Evaluated {chunk_idx + 1} / {len(data)}')
    return study


def _process_file(dry, *args, out=None, **kwargs):
    if dry:
        out = aux.ConsoleOutput.get(out)
        out.write(f'{_process_file.__name__}: {json.dumps(kwargs)}')
        return None, {}
    else:
        return __process_file(*args, out=out, **kwargs)


def __process_file(pipeline, data, im_filepath, seg_filepath, seg_border, log_filepath, adj_filepath, config, first_stage, last_stage, out=None):
    if seg_filepath is not None: aux.mkdir(pathlib.Path(seg_filepath).parents[0])
    if adj_filepath is not None: aux.mkdir(pathlib.Path(adj_filepath).parents[0])
    aux.mkdir(pathlib.Path(log_filepath).parents[0])

    g_raw = io.imread(im_filepath)
    out   = aux.ConsoleOutput.get(out)

    def write_adjacencies_image(name, data):
        if adj_filepath is not None:
            img = render.render_adjacencies(data, override_img=render.render_ymap(data), edge_color=(0,1,0), endpoint_color=(0,1,0))
            io.imwrite(adj_filepath, img)

    atomic_stage = pipeline.stages[pipeline.find('atoms')]
    atomic_stage.add_callback('end', write_adjacencies_image)
    result_data, _, timings = pipeline.process_image(g_raw, data=data, cfg=config, first_stage=first_stage, last_stage=last_stage, log_root_dir=log_filepath, out=out)
    atomic_stage.remove_callback('end', write_adjacencies_image)

    if seg_filepath is not None:
        if seg_border is None: seg_border = 8
        im_result = render.render_postprocessed_result(result_data, seg_border=seg_border)
        aux.mkdir(pathlib.Path(seg_filepath).parents[0])
        io.imwrite(seg_filepath, im_result)
    return result_data, timings


def find_first_differing_stage(pipeline, config1, config2):
    stage_names = [stage.name for stage in pipeline.stages]
    for stage_name in stage_names:
        if      (stage_name in config1 and stage_name not in config2) or \
                (stage_name not in config1 and stage_name in config2) or \
                (stage_name in config1 and stage_name in config2 and config1[stage_name] != config2[stage_name]):
            return stage_name
    return ''


def _resolve_timings_key(key, candidates):
    for c in candidates:
        if str(c) == key: return c
    raise ValueError(f'cannot resolve key "{key}"')


class Task:
    def __init__(self, path, data, parent_task=None, rel_path=None):
        self.runnable    = 'runnable' in data and bool(data['runnable']) == True
        self.parent_task = parent_task
        self.rel_path    = rel_path
        self.path = path
        self.data = data if parent_task is None else config.derive(parent_task.data, data)
        if self.runnable:
            self.   im_pathpattern = os.path.expanduser(self.data['im_pathpattern'])
            self.   gt_pathpattern = os.path.expanduser(self.data['gt_pathpattern'])
            self.     gt_is_unique = self.data['gt_is_unique']
            self.        gt_loader = self.data['gt_loader']
            self. gt_loader_kwargs = self.data['gt_loader_kwargs'] if 'gt_loader_kwargs' in self.data else {}
            self.  seg_pathpattern = path / self.data['seg_pathpattern'] if 'seg_pathpattern' in self.data else None
            self.  adj_pathpattern = path / self.data['adj_pathpattern'] if 'adj_pathpattern' in self.data else None
            self.  log_pathpattern = path / self.data['log_pathpattern']
            self.         file_ids = sorted(frozenset(self.data['file_ids']))
            self.      result_path = path / 'data.dill.gz'
            self.       study_path = path / 'study.csv'
            self.     timings_path = path / 'timings.csv'
            self.timings_json_path = path / '.timings.json'
            self.      digest_path = path / '.digest'
            self.  digest_cfg_path = path / '.digest.cfg.json'
            self.           config = self.data['config']
            self.       seg_border = self.data['seg_border'] if 'seg_border' in self.data else None
            self.           dilate = self.data['dilate']
            self.  merge_threshold = self.data['merge_overlap_threshold']
            self.       last_stage = self.data['last_stage'] if 'last_stage' in self.data else None
            self.          environ = self.data['environ'] if 'environ' in self.data else {}

    def _fmt_path(self, path):
        if self.rel_path is None: return str(path)
        else: return str(path.relative_to(self.rel_path))

    def _initialize(self):
        for key, val in self.environ.items():
            os.environ[key] = str(val)
        ray.init(num_cpus=self.data['num_cpus'], log_to_driver=False, logging_level=ray.logging.ERROR)
        _pipeline = pipeline.create_default_pipeline()
        return _pipeline

    def _shutdown(self):
        ray.shutdown()

    def _load_timings(self):
        if self.timings_json_path.exists():
            with self.timings_json_path.open('r') as fin:
                timings = json.load(fin)
            return {_resolve_timings_key(key, self.file_ids): timings[key] for key in timings}
        else:
            return {}

    def run(self, run_count=1, dry=False, verbosity=0, force=False, one_shot=False, fast_evaluation=False, print_study=False, out=None):
        out = aux.ConsoleOutput.get(out)
        if not self.runnable: return
        config_digest = hashlib.md5(json.dumps(self.config).encode('utf8')).hexdigest()
        if not force and self.digest_path.exists() and self.digest_path.read_text() == config_digest:
            out.write(f'\nSkipping task: {self._fmt_path(self.path)} ({run_count})')
            return
        out.write(aux.Text.style(f'\nEntering task: {self._fmt_path(self.path)} ({run_count})', aux.Text.YELLOW))
        out2 = out.derive(margin=2)
        pipeline = self._initialize()
        try:
            first_stage, data = self.find_first_stage_name(pipeline, dry, out=out2)
            out3 = out2.derive(margin=2, muted = (verbosity <= -int(not dry)))
            timings = self._load_timings()
            discarded_workloads = []
            for file_idx, file_id in enumerate(self.file_ids):
                im_filepath = str(self. im_pathpattern) % file_id
                progress    = file_idx / len(self.file_ids)
                out3.write(aux.Text.style(f'\nProcessing file: {im_filepath}', aux.Text.BOLD) + f' ({100 * progress:.0f}%)')
                kwargs = dict( im_filepath = im_filepath,
                              seg_filepath = str(self.seg_pathpattern) % file_id if self.seg_pathpattern is not None else None,
                              adj_filepath = str(self.adj_pathpattern) % file_id if self.adj_pathpattern is not None else None,
                              log_filepath = str(self.log_pathpattern) % file_id,
                                seg_border = self.seg_border,
                                last_stage = self.last_stage,
                                    config = config.derive(self.config, {}))
                if file_id not in data: data[file_id] = None
                if self.last_stage is not None and pipeline.find(self.last_stage) < pipeline.find('postprocess'): kwargs['seg_filepath'] = None
                data[file_id], _timings = _process_file(dry, pipeline, data[file_id], first_stage=first_stage, out=out3, **kwargs)
                if file_id not in timings: timings[file_id] = {}
                timings[file_id].update(_timings)
                if not dry: discarded_workloads.append(aux.get_discarded_workload(data[file_id]))
            out2.write('')
            if not dry and len(discarded_workloads) > 0:
                out2.write(aux.Text.style('Discarded workload: ', aux.Text.BOLD) + f'{100 * min(discarded_workloads):.1f}% – {100 * max(discarded_workloads):.1f}% (avg {100 * np.mean(discarded_workloads):.1f}% ±{100 * np.std(discarded_workloads):.1f})')
            if one_shot or ((first_stage is not None and pipeline.find(first_stage) >= pipeline.find('postprocess') or (self.last_stage is not None and pipeline.find(self.last_stage) <= pipeline.find('atoms'))) and not self.result_path.exists()):
                out2.write('Skipping writing results')
            else:
                if not dry:
                    self.write_timings(timings)
                    out2.intermediate(f'Writing results... {self._fmt_path(self.result_path)}')
                    with gzip.open(self.result_path, 'wb') as fout:
                        dill.dump(data, fout, byref=True)
                    with self.digest_cfg_path.open('w') as fout:
                        json.dump(self.config, fout)
                out2.write(aux.Text.style('Results written to: ', aux.Text.BOLD) + self._fmt_path(self.result_path))
            if self.last_stage is not None and pipeline.find(self.last_stage) < pipeline.find('postprocess'):
                out2.write('Skipping evaluation')
            else:
                if not dry:
                    shallow_data = {file_id : {key : data[file_id][key] for key in ('g_raw', 'postprocessed_candidates')} for file_id in self.file_ids}
                    del data
                    study = evaluate(shallow_data, self.gt_pathpattern, self.gt_is_unique, self.gt_loader, self.gt_loader_kwargs, dict(merge_overlap_threshold=self.merge_threshold, dilate=self.dilate), fast=fast_evaluation, out=out2)
                    self.write_evaluation_results(shallow_data.keys(), study)
                    if not one_shot: self.digest_path.write_text(config_digest)
                out2.write(aux.Text.style('Evaluation study written to: ', aux.Text.BOLD) + self._fmt_path(self.study_path))
                if not dry and print_study:
                    out2.write('')
                    study.print_results(write=out2.write, line_suffix='', pad=2)
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

    def find_pickup_candidates(self, pipeline):
        pickup_candidates = []
        previous_task = self.find_parent_task_with_result()
        if previous_task is not None:
            first_stage = find_first_differing_stage(pipeline, self.config, previous_task.config)
            pickup_candidates.append((previous_task, first_stage))
        if self.result_path.exists() and self.digest_cfg_path.exists():
            with self.digest_cfg_path.open('r') as fin:
                config = json.load(fin)
            first_stage = find_first_differing_stage(pipeline, self.config, config)
            pickup_candidates.append((self, first_stage))
        return pickup_candidates

    def find_best_pickup_candidate(self, pipeline):
        pickup_candidates = self.find_pickup_candidates(pipeline)
        if len(pickup_candidates) == 0: return None, None
        pickup_candidate_scores = [pipeline.find(first_stage) for task, first_stage in pickup_candidates]
        return pickup_candidates[np.argmax(pickup_candidate_scores)]

    def find_first_stage_name(self, pipeline, dry=False, out=None):
        out = aux.ConsoleOutput.get(out)
        pickup_task, stage_name = self.find_best_pickup_candidate(pipeline)
        if pickup_task is None or pipeline.find(stage_name) <= pipeline.find('atoms') + 1:
            return None, {}
        else:
            out.write(f'Picking up from: {self._fmt_path(pickup_task.result_path)} ({stage_name if stage_name != "" else "evaluate"})')
            if not dry:
                with gzip.open(pickup_task.result_path, 'rb') as fin:
                    data = dill.load(fin)
                return stage_name, data
            else:
                return stage_name, {}

    def write_evaluation_results(self, chunk_ids, study):
        measure_names = sorted(study.measures.keys())
        rows = [[str(self.path)], ['ID'] + measure_names]
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

    def write_timings(self, timings):
        file_ids = timings.keys()
        stage_names = sorted(list(timings.values())[0].keys())
        rows = [[str(self.path)], ['ID'] + stage_names + ['total']]
        totals = np.zeros(len(stage_names) + 1)
        for file_id in file_ids:
            vals  = [timings[file_id][stage_name] for stage_name in stage_names]
            vals += [sum(vals)]
            row   = [file_id] + [_format_runtime(val) for val in vals]
            rows.append(row)
            totals += np.asarray(vals)
        rows.append([''] + [_format_runtime(val) for val in totals])
        with self.timings_path.open('w', newline='') as fout:
            csv_writer = csv.writer(fout, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in rows:
                csv_writer.writerow(row)
        with self.timings_json_path.open('w') as fout:
            json.dump(timings, fout)


class BatchLoader:

    def __init__(self, override_cfg={}):
        self.tasks        = []
        self.override_cfg = override_cfg

    def load(self, path):
        root_path = pathlib.Path(path)
        self.process_directory(root_path, rel_path=root_path.parents[0])

    def process_directory(self, current_dir, parent_task=None, rel_path=None):
        task_file = current_dir / 'task.json'
        if task_file.exists():
            try:
                with task_file.open('r') as task_fin:
                    task_data = json.load(task_fin)
                task = Task(current_dir, task_data, parent_task, rel_path=rel_path)
                for key in self.override_cfg:
                    setattr(task, key, self.override_cfg[key])
            except json.JSONDecodeError as err:
                raise ValueError(f'Error processing: "{task_file}"')
            self.tasks.append(task)
            parent_task = task
        for d in os.listdir(current_dir):
            f = current_dir / d
            if f.is_dir():
                self.process_directory(f, parent_task, rel_path=rel_path)


def get_path(root_path, path):
    if isinstance(root_path, str): root_path = pathlib.Path(root_path)
    if isinstance(     path, str):      path = pathlib.Path(     path)
    if path.is_absolute(): return path
    return pathlib.Path(root_path) / path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='root directory for batch processing')
    parser.add_argument('--run', help='run batch processing', action='store_true')
    parser.add_argument('--verbosity', help='postive (negative) is more (less) verbose', type=int, default=0)
    parser.add_argument('--force', help='do not skip tasks', action='store_true')
    parser.add_argument('--oneshot', help='do not save results or mark tasks as processed', action='store_true')
    parser.add_argument('--fast-evaluation', help='only use fast measures for evaluation', action='store_true')
    parser.add_argument('--last-stage', help='override the "last_stage" setting', type=str, default=None)
    parser.add_argument('--print-study', help='print out evaluation results', action='store_true')
    parser.add_argument('--task', help='run only the given task', type=str, default=[], action='append')
    parser.add_argument('--task-dir', help='run only the given task and those from its sub-directories', type=str, default=[], action='append')
    args = parser.parse_args()

    if args.fast_evaluation and not args.oneshot:
        parser.error('Using "--fast-evaluation" only allowed if "--oneshot" is used')

    if args.last_stage is not None and not args.oneshot:
        parser.error('Using "--last-stage" only allowed if "--oneshot" is used')

    loader = BatchLoader(override_cfg=dict(last_stage=args.last_stage))
    loader.load(args.path)

    args.task     = [get_path(args.path,     task_path) for     task_path in args.task    ]
    args.task_dir = [get_path(args.path, task_dir_path) for task_dir_path in args.task_dir]

    dry = not args.run
    out = aux.ConsoleOutput()
    runnable_tasks = [task for task in loader.tasks if task.runnable]
    run_task_count = 0
    out.write(f'Loaded {len(runnable_tasks)} runnable task(s)')
    if dry: out.write(f'DRY RUN: use "--run" to run the tasks instead')
    for task in runnable_tasks:
        if (len(args.task) > 0 or len(args.task_dir) > 0) and all(task.path != path for path in args.task) and all(not aux.is_subpath(path, task.path) for path in args.task_dir): continue
        run_task_count += 1
        newpid = os.fork()
        if newpid == 0:
            task.run(run_task_count, dry, args.verbosity, args.force, args.oneshot, args.fast_evaluation, args.print_study, out)
            os._exit(0)
        else:
            if os.waitpid(newpid, 0)[1] != 0:
                out.write('An error occurred: interrupting')
                sys.exit(1)
    out.write(f'\nRan {run_task_count} task(s)')


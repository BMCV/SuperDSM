from .config import get_config_value, derive_config
from .pipeline import create_default_pipeline
from .candidates import _process_candidates
from ._aux import get_output, mkdir, Text, get_discarded_workload, is_subpath
from .io import imread, imwrite
from .render import rasterize_labels, render_ymap, render_atoms, render_adjacencies, render_result_over_image
from .automation import create_config

import sys, os, pathlib, json, gzip, dill, tempfile, subprocess, skimage, warnings, csv, hashlib, tarfile, shutil, time, itertools, re
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
        img = skimage.io.imread(png_file.name, plugin='matplotlib', format='png')[:,:,3]
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
        return imread(filepath)
    elif loader == 'xcf':
        return load_unlabeled_xcf_gt(filepath, **loader_kwargs)


FP_INVARIANT_MEASURES = frozenset(['SEG', 'd/Split', 'd/Merge', 'd/FN'])


def evaluate(data, gt_pathpattern, gt_is_unique, gt_loader, gt_loader_kwargs, rasterize_kwargs, out=None):
    out = get_output(out)
    import segmetrics

    segmetrics.detection.FalseMerge.ACCUMULATIVE    = False
    segmetrics.detection.FalseSplit.ACCUMULATIVE    = False
    segmetrics.detection.FalsePositive.ACCUMULATIVE = False
    segmetrics.detection.FalseNegative.ACCUMULATIVE = False

    study = segmetrics.study.Study()
    study.add_measure(segmetrics. regional.Dice()         , 'Dice')
    study.add_measure(segmetrics. regional.JaccardIndex() , 'Jaccard')
    study.add_measure(segmetrics. regional.RandIndex()    , 'Rand')
    study.add_measure(segmetrics. regional.ISBIScore()    , 'SEG')
    study.add_measure(segmetrics.detection.FalseSplit()   , 'd/Split')
    study.add_measure(segmetrics.detection.FalseMerge()   , 'd/Merge')
    study.add_measure(segmetrics.detection.FalsePositive(), 'd/FP')
    study.add_measure(segmetrics.detection.FalseNegative(), 'd/FN')
    study.add_measure(segmetrics.boundary.ObjectBasedDistance(segmetrics.boundary.NSD()           , skip_fn=False), 'NSD')
    study.add_measure(segmetrics.boundary.ObjectBasedDistance(segmetrics.boundary.Hausdorff('a2e'), skip_fn=False), 'HSD (a2e)')
    study.add_measure(segmetrics.boundary.ObjectBasedDistance(segmetrics.boundary.Hausdorff('e2a'), skip_fn=False), 'HSD (e2a)')
    study.add_measure(segmetrics.boundary.ObjectBasedDistance(segmetrics.boundary.NSD()           , skip_fn=True ), 'NSD*')
    study.add_measure(segmetrics.boundary.ObjectBasedDistance(segmetrics.boundary.Hausdorff('a2e'), skip_fn=True ), 'HSD* (a2e)')
    study.add_measure(segmetrics.boundary.ObjectBasedDistance(segmetrics.boundary.Hausdorff('e2a'), skip_fn=True ), 'HSD* (e2a)')

    chunk_ids = sorted(data.keys())
    for chunk_idx, chunk_id in enumerate(chunk_ids):
        actual = rasterize_labels(data[chunk_id], **rasterize_kwargs)
        expected = load_gt(gt_loader, filepath=gt_pathpattern % chunk_id, **gt_loader_kwargs)
        study.set_expected(expected, unique=gt_is_unique)
        study.process(actual, unique=True, chunk_id=chunk_id)
        out.intermediate(f'Evaluated {chunk_idx + 1} / {len(data)}')
    return study


def _process_file(dry, *args, out=None, **kwargs):
    if dry:
        out = get_output(out)
        out.write(f'{_process_file.__name__}: {json.dumps(kwargs)}')
        return None, {}
    else:
        return __process_file(*args, out=out, **kwargs)


def __process_file(pipeline, data, im_filepath, seg_filepath, seg_border, log_filepath, adj_filepath, cfg_filepath, cfg, first_stage, last_stage, out=None):
    if seg_filepath is not None: mkdir(pathlib.Path(seg_filepath).parents[0])
    if adj_filepath is not None: mkdir(pathlib.Path(adj_filepath).parents[0])
    if log_filepath is not None: mkdir(pathlib.Path(log_filepath).parents[0])
    if cfg_filepath is not None: mkdir(pathlib.Path(cfg_filepath).parents[0])

    histological  = get_config_value(cfg, 'histological', False)
    imread_kwargs = {}
    if histological:
        imread_kwargs['as_gray'] = False

    g_raw = imread(im_filepath, **imread_kwargs)
    out   = get_output(out)

    timings = {}
    if first_stage != '':
        out.intermediate('Creating configuration...')
        t0 = time.time()
        if histological:
            g_gray = g_raw.mean(axis=2)
            g_gray = g_gray.max() - g_gray
        else:
            g_gray = g_raw
        cfg, scale = create_config(cfg, g_gray)
        timings['autocfg'] = time.time() - t0
        with open(cfg_filepath, 'w') as fout:
            json.dump(cfg, fout)
        if scale is not None:
            out.write(f'Estimated scale: {scale:.2f}')

    def write_adjacencies_image(name, data):
        if adj_filepath is not None:
            ymap = render_ymap(data)
            ymap = render_atoms(data, override_img=ymap, border_color=(0,0,0), border_radius=1)
            img  = render_adjacencies(data, override_img=ymap, edge_color=(0,1,0), endpoint_color=(0,1,0))
            imwrite(adj_filepath, img)

    atomic_stage = pipeline.stages[pipeline.find('top-down-segmentation')]
    atomic_stage.add_callback('end', write_adjacencies_image)
    result_data, _, _timings = pipeline.process_image(g_raw, data=data, cfg=cfg, first_stage=first_stage, last_stage=last_stage, log_root_dir=log_filepath, out=out)
    atomic_stage.remove_callback('end', write_adjacencies_image)
    timings.update(_timings)

    if seg_filepath is not None:
        if seg_border is None: seg_border = 8
        im_result = render_result_over_image(result_data, border_width=seg_border)
        mkdir(pathlib.Path(seg_filepath).parents[0])
        imwrite(seg_filepath, im_result)
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


def _find_task_rel_path(task):
    if task.parent_task is not None:
        return _find_task_rel_path(task.parent_task)
    else:
        return task.path.parents[0]


def _compress_logs(log_dir):
    if log_dir is None: return
    log_dir_path = pathlib.Path(log_dir)
    if not log_dir_path.exists(): return
    assert log_dir_path.is_dir()
    compressed_logs_filepath = f'{log_dir}.tgz'
    with tarfile.open(compressed_logs_filepath, 'w:gz') as tar:
        tar.add(log_dir, arcname=os.path.sep)
    shutil.rmtree(str(log_dir))


DATA_DILL_GZ_FILENAME = 'data.dill.gz'


class Task:
    def __init__(self, path, data, parent_task=None, rel_path=None):
        self.runnable    = 'runnable' in data and bool(data['runnable']) == True
        self.parent_task = parent_task
        self.path = path
        self.data = data if parent_task is None else derive_config(parent_task.data, data)
        self.rel_path = _find_task_rel_path(self)
        self.file_ids = sorted(frozenset(self.data['file_ids'])) if 'file_ids' in self.data else None
        self.fully_annotated_ids = frozenset(self.data['fully_annotated_ids']) if 'fully_annotated_ids' in self.data else self.file_ids
        self.     im_pathpattern = os.path.expanduser(self.data['im_pathpattern']) if 'im_pathpattern' in self.data else None
        self.     gt_pathpattern = os.path.expanduser(self.data['gt_pathpattern']) if 'gt_pathpattern' in self.data else None
        self.       gt_is_unique = self.data.get('gt_is_unique'    , None)
        self.          gt_loader = self.data.get('gt_loader'       , None)
        self.   gt_loader_kwargs = self.data.get('gt_loader_kwargs', {}  )

        if 'base_config_path' in self.data:
            base_config_path = self.data['base_config_path']
            base_config_path = pathlib.Path(base_config_path.replace('{DIRNAME}', path.name).replace('{ROOTDIR}', str(self.root_path)))
            if not base_config_path.is_absolute():
                base_config_path = path / base_config_path
            with base_config_path.open('r') as base_config_fin:
                base_config = json.load(base_config_fin)
            parent_config = get_config_value(parent_task.data, 'config', {})
            self.data['config'] = derive_config(derive_config(parent_config, base_config), get_config_value(data, 'config', {}))
            del self.data['base_config_path']

        if self.runnable:

            assert self.file_ids            is not None
            assert self.fully_annotated_ids is not None
            assert self.im_pathpattern      is not None
            assert self.gt_pathpattern      is not None
            assert self.gt_is_unique        is not None
            assert self.gt_loader           is not None
            assert self.gt_loader_kwargs    is not None

            self.  seg_pathpattern = path / self.data['seg_pathpattern'] if 'seg_pathpattern' in self.data else None
            self.  adj_pathpattern = path / self.data['adj_pathpattern'] if 'adj_pathpattern' in self.data else None
            self.  log_pathpattern = path / self.data['log_pathpattern'] if 'log_pathpattern' in self.data else None
            self.  cfg_pathpattern = path / self.data['cfg_pathpattern'] if 'cfg_pathpattern' in self.data else None
            self.      result_path = path / DATA_DILL_GZ_FILENAME
            self.       study_path = path / 'study.csv'
            self.     timings_path = path / 'timings.csv'
            self.timings_json_path = path / '.timings.json'
            self.      digest_path = path / '.digest'
            self.  digest_cfg_path = path / '.digest.cfg.json'
            self. fn_analysis_path = path / 'fn.csv'
            self.           config = self.data['config']
            self.       seg_border = self.data['seg_border'] if 'seg_border' in self.data else None
            self.           dilate = self.data['dilate']
            self.  merge_threshold = self.data['merge_overlap_threshold']
            self.       last_stage = self.data['last_stage'] if 'last_stage' in self.data else None
            self.          environ = self.data['environ'] if 'environ' in self.data else {}

    @staticmethod
    def create_from_directory(task_dir, parent_task, override_cfg={}, force_runnable=False):
        task_file = task_dir / 'task.json'
        if task_file.exists():
            try:
                with task_file.open('r') as task_fin:
                    task_data = json.load(task_fin)
                if force_runnable: task_data['runnable'] = True
                task = Task(task_dir, task_data, parent_task)
                for key in override_cfg:
                    setattr(task, key, override_cfg[key])
                return task
            except json.JSONDecodeError as err:
                raise ValueError(f'Error processing: "{task_file}"')
        return None
    
    @property
    def root_path(self):
        if self.parent_task is not None: return self.parent_task.root_path
        else: return self.path

    def _fmt_path(self, path):
        if isinstance(path, str): path = pathlib.Path(path)
        if self.rel_path is None: return str(path)
        else: return str(path.relative_to(self.rel_path))

    def _initialize(self):
        for key, val in self.environ.items():
            os.environ[key] = str(val)
        ray.init(num_cpus=self.data['num_cpus'], log_to_driver=False, logging_level=ray.logging.ERROR)
        _pipeline = create_default_pipeline()
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
        
    @property
    def config_digest(self):
        return hashlib.md5(json.dumps(self.config).encode('utf8')).hexdigest()
        
    @property
    def is_pending(self):
        return self.runnable and not (self.digest_path.exists() and self.digest_path.read_text() == self.config_digest)

    def run(self, task_info=None, dry=False, verbosity=0, force=False, one_shot=False, evaluation='full', print_study=False, debug=False, report=None, out=None):
        assert evaluation in ('none', 'legacy', 'full')
        out = get_output(out)
        if not self.runnable: return
        _process_candidates._DEBUG = debug
        if not force and not self.is_pending:
            out.write(f'\nSkipping task: {self._fmt_path(self.path)} {"" if task_info is None else f"({task_info})"}')
            return
        if self.last_stage is not None:
            if task_info is not None: task_info = f'{task_info}, '
            else: task_info = ''
            task_info = task_info + f'last stage: {self.last_stage}'
        out.write(Text.style(f'\nEntering task: {self._fmt_path(self.path)} {"" if task_info is None else f"({task_info})"}', Text.YELLOW))
        out2 = out.derive(margin=2)
        pipeline = self._initialize()
        assert self.last_stage is None or self.last_stage == '' or not np.isinf(pipeline.find(self.last_stage)), f'unknown stage "{self.last_stage}"'
        try:
            first_stage, data = self.find_first_stage_name(pipeline, dry, out=out2)
            out3 = out2.derive(margin=2, muted = (verbosity <= -int(not dry)))
            timings = self._load_timings()
            discarded_workloads = []
            for file_idx, file_id in enumerate(self.file_ids):
                im_filepath = str(self. im_pathpattern) % file_id
                progress    = file_idx / len(self.file_ids)
                if report is not None: report.update(self, progress)
                out3.write(Text.style(f'\nProcessing file: {im_filepath}', Text.BOLD) + f' ({100 * progress:.0f}%)')
                kwargs = dict( im_filepath = im_filepath,
                              seg_filepath = str(self.seg_pathpattern) % file_id if self.seg_pathpattern is not None else None,
                              adj_filepath = str(self.adj_pathpattern) % file_id if self.adj_pathpattern is not None else None,
                              log_filepath = str(self.log_pathpattern) % file_id if self.log_pathpattern is not None else None,
                              cfg_filepath = str(self.cfg_pathpattern) % file_id if self.cfg_pathpattern is not None else None,
                                seg_border = self.seg_border,
                                last_stage = self.last_stage,
                                       cfg = derive_config(self.config, {}))
                if file_id not in data: data[file_id] = None
                if self.last_stage is not None and pipeline.find(self.last_stage) < pipeline.find('postprocess'): kwargs['seg_filepath'] = None
                data[file_id], _timings = _process_file(dry, pipeline, data[file_id], first_stage=first_stage, out=out3, **kwargs)
                if not dry: _compress_logs(kwargs['log_filepath'])
                if file_id not in timings: timings[file_id] = {}
                timings[file_id].update(_timings)
                if not dry and 'candidates' in data[file_id]:
                    discarded_workload = get_discarded_workload(data[file_id])
                    if not np.isnan(discarded_workload): discarded_workloads.append(discarded_workload)
            out2.write('')
            if report is not None: report.update(self, 'active')
            if not dry and len(discarded_workloads) > 0:
                out2.write(Text.style('Discarded workload: ', Text.BOLD) + f'{100 * min(discarded_workloads):.1f}% – {100 * max(discarded_workloads):.1f}% (avg {100 * np.mean(discarded_workloads):.1f}% ±{100 * np.std(discarded_workloads):.1f})')
            
            skip_writing_results_conditions = [
                one_shot,
                self.last_stage is not None and pipeline.find(self.last_stage) <= pipeline.find('modelfit') and not self.result_path.exists(),
                first_stage is not None and pipeline.find(first_stage) >= pipeline.find('postprocess')
            ]
            skip_evaluation = (self.last_stage is not None and pipeline.find(self.last_stage) < pipeline.find('postprocess'))
            if any(skip_writing_results_conditions):
                out2.write('Skipping writing results')
            else:
                if not dry:
                    self.write_timings(timings)
                    out2.intermediate(f'Writing results... {self._fmt_path(self.result_path)}')
                    with gzip.open(self.result_path, 'wb') as fout:
                        dill.dump(data, fout, byref=True)
                    with self.digest_cfg_path.open('w') as fout:
                        json.dump(self.config, fout)
                    if not one_shot and skip_evaluation: self.digest_path.write_text(self.config_digest)
                out2.write(Text.style('Results written to: ', Text.BOLD) + self._fmt_path(self.result_path))
            if evaluation == 'none' or skip_evaluation:
                out2.write('Skipping evaluation')
            else:
                if not dry:
                    shallow_data = {file_id : {key : data[file_id][key] for key in ('g_raw', 'postprocessed_candidates')} for file_id in self.file_ids}
                    del data
                    out2.intermediate('Evaluating...')
                    study = evaluate(shallow_data, self.gt_pathpattern, self.gt_is_unique, self.gt_loader, self.gt_loader_kwargs, dict(merge_overlap_threshold=self.merge_threshold, dilate=self.dilate), out=out2)
                    self.write_evaluation_results(shallow_data.keys(), study)
                    if not one_shot: self.digest_path.write_text(self.config_digest)
                out2.write(Text.style('Evaluation study written to: ', Text.BOLD) + self._fmt_path(self.study_path))
                if not dry and print_study:
                    out2.write('')
                    study.print_results(write=out2.write, line_suffix='', pad=2)
            for obj_name in ('data', 'shallow_data'):
                if obj_name in locals(): return locals()[obj_name]
        except:
            out.write(Text.style(f'\nError while processing task: {self._fmt_path(self.path)}', Text.RED))
            raise
        finally:
            self._shutdown()

    def analyze_fn(self, dry=False, pp_logfilename='postprocessing.txt', out=None):
        out = get_output(out)
        out = out.derive(margin=2)
        if not self.runnable or dry: return
        if self.log_pathpattern is None: return
        log_path_parent = self.log_pathpattern.parent
        while not log_path_parent.is_dir():
            log_path_parent = log_path_parent.parent
        if self.fn_analysis_path.is_file() and self.fn_analysis_path.stat().st_mtime >= log_path_parent.stat().st_mtime: return
        pp_logfile_line_pattern = re.compile(r'^object at x=([0-9]+), y=([0-9]+): ([^(]+).*?$')
        reasons_histogram = {}
        for file_id in self.file_ids:
            log_filepath = str(self.log_pathpattern) % file_id
            compressed_logs_filepath = f'{log_filepath}.tgz'
            out.intermediate(f'Analyzing false-negative detections for file: {file_id}')
            expected = load_gt(self.gt_loader, filepath=self.gt_pathpattern % file_id, **self.gt_loader_kwargs)
            if not dry:
                with tarfile.open(compressed_logs_filepath, 'r:gz') as tar:
                    if not pp_logfilename in tar.getnames(): continue
                    pp_logfile_info = tar.getmember(pp_logfilename)
                    pp_logfile = tar.extractfile(pp_logfile_info)
                    pp_logfile_text = pp_logfile.read().decode('utf-8')
                    for pp_logfile_line in pp_logfile_text.split('\n'):
                        if len(pp_logfile_line) == 0: continue
                        match = pp_logfile_line_pattern.match(pp_logfile_line)
                        if match is None: raise ValueError(f'Log file {compressed_logs_filepath} contains malformed line: {pp_logfile_line}')
                        x = int(match.group(1))
                        y = int(match.group(2))
                        reason = match.group(3)
                        y = np.clip(y, 0, expected.shape[0] - 1)
                        x = np.clip(x, 0, expected.shape[1] - 1)
                        is_fn = (expected[y,x] > 0)
                        if is_fn:
                            reasons_histogram[reason] = reasons_histogram.get(reason, 0) + 1
        with self.fn_analysis_path.open('w', newline='') as fout:
            csv_writer = csv.writer(fout, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for reason, count in reasons_histogram.items():
                csv_writer.writerow([reason, count])

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
        out = get_output(out)
        pickup_task, stage_name = self.find_best_pickup_candidate(pipeline)
        if pickup_task is None or pipeline.find(stage_name) <= pipeline.find('modelfit') + 1:
            return None, {}
        else:
            out.write(f'Picking up from: {self._fmt_path(pickup_task.result_path)} ({stage_name if stage_name != "" else "load"})')
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
            chunks  = study. results[measure_name]
            values  = list(itertools.chain(*[chunks[chunk_id] for chunk_id in chunks if chunk_id in self.fully_annotated_ids or measure_name in FP_INVARIANT_MEASURES]))
            fnc = np.sum if measure.ACCUMULATIVE else np.mean
            val = fnc(values)
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
        self.process_directory(root_path)

    def process_directory(self, current_dir, parent_task=None):
        task = Task.create_from_directory(current_dir, parent_task, self.override_cfg)
        if task is not None:
            self.tasks.append(task)
            parent_task = task
        for d in os.listdir(current_dir):
            f = current_dir / d
            if f.is_dir():
                self.process_directory(f, parent_task)


def get_path(root_path, path):
    if isinstance(root_path, str): root_path = pathlib.Path(root_path)
    if isinstance(     path, str):      path = pathlib.Path(     path)
    if path.is_absolute(): return path
    return pathlib.Path(root_path) / path


class StatusReport:
    def __init__(self, scheduled_tasks, filepath=None):
        self.scheduled_tasks = scheduled_tasks
        self.filepath        = filepath
        self.status          = dict()
        self.task_progress   = None
        
    def get_task_status(self, task):
        return self.status.get(str(task.path), 'skipped')
    
    def update(self, task, status, save=True):
        if isinstance(status, float):
            self.task_progress = status
            status = 'active'
        else:
            self.task_progress = None
        assert status in ('pending', 'done', 'active', 'error')
        if status in ('done', 'active') and self.get_task_status(task) == 'skipped': return
        self.status[str(task.path)] = status
        if save: self.save()
    
    def save(self):
        if self.filepath is None: return
        with open(str(self.filepath), 'w') as fout:
            skipped_tasks = []
            for task in self.scheduled_tasks:
                status = self.get_task_status(task)
                prefix, suffix = '', ''
                if status == 'skipped':
                    skipped_tasks.append(task)
                    continue
                elif status == 'pending': prefix = ' o '
                elif status ==    'done': prefix = ' ✓ '
                elif status ==  'active': prefix = '-> '
                elif status ==   'error': prefix = 'EE '
                if status == 'active' and self.task_progress is not None:
                    suffix = f' ({100 * self.task_progress:.0f}%)'
                fout.write(f'{prefix}{task.path}{suffix}\n')
            if len(skipped_tasks) > 0:
                fout.write('\nSkipped tasks:\n')
                for task in skipped_tasks:
                    fout.write(f'- {str(task.path)}\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='root directory for batch processing')
    parser.add_argument('--run', help='run batch processing', action='store_true')
    parser.add_argument('--verbosity', help='postive (negative) is more (less) verbose', type=int, default=0)
    parser.add_argument('--force', help='do not skip tasks', action='store_true')
    parser.add_argument('--oneshot', help='do not save results or mark tasks as processed', action='store_true')
    parser.add_argument('--last-stage', help='override the "last_stage" setting', type=str, default=None)
    parser.add_argument('--print-study', help='print out evaluation results', action='store_true')
    parser.add_argument('--skip-evaluation', help='skips evaluation', action='store_true')
    parser.add_argument('--task', help='run only the given task', type=str, default=[], action='append')
    parser.add_argument('--task-dir', help='run only the given task and those from its sub-directories', type=str, default=[], action='append')
    parser.add_argument('--debug', help='do not use multiprocessing', action='store_true')
    parser.add_argument('--analyze-fn', help='summarize reasons of false negative detections', action='store_true')
    parser.add_argument('--report', help='report current status to file', type=str, default='/tmp/godmod-status')
    args = parser.parse_args()

    if args.last_stage is not None and not args.oneshot:
        parser.error('Using "--last-stage" only allowed if "--oneshot" is used')

    override_cfg = dict()
    if args.last_stage is not None:
        override_cfg['last_stage'] = args.last_stage
        
    loader = BatchLoader(override_cfg=override_cfg)
    loader.load(args.path)

    args.task     = [get_path(args.path,     task_path) for     task_path in args.task    ]
    args.task_dir = [get_path(args.path, task_dir_path) for task_dir_path in args.task_dir]

    dry = not args.run
    out = get_output()
    runnable_tasks = [task for task in loader.tasks if task.runnable]
    out.write(f'Loaded {len(runnable_tasks)} runnable task(s)')
    if dry: out.write(f'DRY RUN: use "--run" to run the tasks instead')
    scheduled_tasks     = []
    run_task_count      =  0
    pending_tasks_count =  0
    report = StatusReport(scheduled_tasks, filepath=None if dry else args.report)
    for task in runnable_tasks:
        if (len(args.task) > 0 or len(args.task_dir) > 0) and all(task.path != path for path in args.task) and all(not is_subpath(path, task.path) for path in args.task_dir): continue
        scheduled_tasks.append(task)
        if task.is_pending or args.force:
            pending_tasks_count += 1
            report.update(task, 'pending', save=False)
    for task in scheduled_tasks:
        if task.is_pending or args.force:
            run_task_count += 1
            task_info = f'{run_task_count} of {pending_tasks_count}'
        else:
            task_info = None
        report.update(task, 'active')
        newpid = os.fork()
        if newpid == 0:
            evaluation = 'none' if args.skip_evaluation else 'full'
            try:
                task.run(task_info, dry, args.verbosity, args.force, args.oneshot, evaluation, args.print_study, args.debug, report, out)
            except:
                report.update(task, 'error')
                raise
            if args.analyze_fn:
                task.analyze_fn(dry, out=out)
            os._exit(0)
        else:
            if os.waitpid(newpid, 0)[1] != 0:
                out.write('An error occurred: interrupting')
                sys.exit(1)
            else:
                report.update(task, 'done')
    out.write(f'\nRan {run_task_count} task(s) out of {len(runnable_tasks)} in total')

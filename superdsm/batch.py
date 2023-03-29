from .pipeline import create_default_pipeline
from .objects import _compute_objects
from ._aux import mkdir, is_subpath, copy_dict
from .output import get_output, Text
from .io import imread, imwrite
from .render import rasterize_labels, render_ymap, render_atoms, render_adjacencies, render_result_over_image
from .automation import create_config
from .config import Config
from .globalenergymin import PerformanceReport

import sys, os, pathlib, json, gzip, dill, tempfile, subprocess, skimage, warnings, csv, tarfile, shutil, time, itertools, re
import ray
import numpy as np
import scipy.ndimage as ndi


def _format_runtime(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


def _resolve_pathpattern(pathpattern, fileid):
    if pathpattern is None: return None
    else: return str(pathpattern) % fileid


def _process_file(dry, *args, out=None, **kwargs):
    if dry:
        out = get_output(out)
        kwargs_serializable = copy_dict(kwargs)
        if 'cfg' in kwargs_serializable:
            kwargs_serializable['cfg'] = kwargs_serializable['cfg'].entries
        out.write(f'{_process_file.__name__}: {json.dumps(kwargs_serializable)}')
        return None, {}
    else:
        return __process_file(*args, out=out, **kwargs)


def __process_file(pipeline, data, img_filepath, overlay_filepath, seg_filepath, seg_border, log_filepath, adj_filepath, cfg_filepath, cfg, first_stage, last_stage, rasterize_kwargs, out=None):
    if     seg_filepath is not None: mkdir(pathlib.Path(    seg_filepath).parents[0])
    if     adj_filepath is not None: mkdir(pathlib.Path(    adj_filepath).parents[0])
    if     log_filepath is not None: mkdir(pathlib.Path(    log_filepath).parents[0])
    if     cfg_filepath is not None: mkdir(pathlib.Path(    cfg_filepath).parents[0])
    if overlay_filepath is not None: mkdir(pathlib.Path(overlay_filepath).parents[0])

    histological  = cfg.get('histological', False)
    imread_kwargs = {}
    if histological:
        imread_kwargs['as_gray'] = False

    g_raw = imread(img_filepath, **imread_kwargs)
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
        cfg, scale = create_config(pipeline, cfg, g_gray)
        timings['autocfg'] = time.time() - t0
        with open(cfg_filepath, 'w') as fout:
            cfg.dump_json(fout)
        if scale is not None:
            out.write(f'Estimated scale: {scale:.2f}')

    def write_adjacencies_image(name, data):
        if adj_filepath is not None:
            ymap = render_ymap(data)
            ymap = render_atoms(data, override_img=ymap, border_color=(0,0,0), border_radius=1)
            img  = render_adjacencies(data, override_img=ymap, edge_color=(0,1,0), endpoint_color=(0,1,0))
            imwrite(adj_filepath, img)

    atomic_stage = pipeline.stages[pipeline.find('c2f-region-analysis')]
    atomic_stage.add_callback('end', write_adjacencies_image)
    result_data, _, _timings = pipeline.process_image(g_raw, data=data, cfg=cfg, first_stage=first_stage, last_stage=last_stage, log_root_dir=log_filepath, out=out)
    atomic_stage.remove_callback('end', write_adjacencies_image)
    timings.update(_timings)

    if overlay_filepath is not None:
        if seg_border is None: seg_border = 8
        img_overlay = render_result_over_image(result_data, border_width=seg_border)
        mkdir(pathlib.Path(overlay_filepath).parents[0])
        imwrite(overlay_filepath, img_overlay)

    if seg_filepath is not None:
        seg_result = rasterize_labels(result_data, **rasterize_kwargs)
        mkdir(pathlib.Path(seg_filepath).parents[0])
        imwrite(seg_filepath, seg_result)

    return result_data, timings


def find_first_differing_stage(pipeline, config1, config2):
    assert isinstance(config1, dict)
    assert isinstance(config2, dict)
    stage_names = [stage.name for stage in pipeline.stages]
    if config1.get('AF_scale', None) != config2.get('AF_scale', None): return stage_names[0]
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


def _write_performance_report(task_path, performance_path, data, overall_performance):
    file_ids = data.keys()
    properties = ['direct_solution_success', 'iterative_pruning_success', 'overall_pruning_success', 'nontrivial_pruning_success']
    fields = PerformanceReport.attributes + properties
    rows = [[str(task_path)], ['ID'] + fields]
    get_row = lambda prefix, performance: [prefix] + [getattr(performance, field) for field in fields]
    for file_id in file_ids:
        row = get_row(str(file_id), data[file_id]['performance'])
        rows.append(row)
    footer_row = get_row('', overall_performance)
    rows.append(footer_row)
    with open(str(performance_path), 'w', newline='') as fout:
        csv_writer = csv.writer(fout, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            csv_writer.writerow(row)


DATA_DILL_GZ_FILENAME = 'data.dill.gz'


class Task:
    """Represents a batch processing task (see :ref:`batch_task_spec`).

    :param path: The path of the directory where the task specification resides.
    :param data: Dictionary corresponding to the task specification (JSON data).
    :param parent_task: The parent task or ``None`` if there is no parent task.
    """

    def __init__(self, path, data, parent_task=None):
        self.runnable    = 'runnable' in data and bool(data['runnable']) == True
        self.parent_task = parent_task
        self.path = path
        self.data = Config(data) if parent_task is None else Config(parent_task.data).derive(data)
        self.rel_path = _find_task_rel_path(self)
        self.file_ids = sorted(frozenset(self.data.entries['file_ids'])) if 'file_ids' in self.data else None
        self.img_pathpattern = self.data.update('img_pathpattern', lambda img_pathpattern: str(self.resolve_path(img_pathpattern)))

        if 'base_config_path' in self.data:
            base_config_path = self.resolve_path(self.data['base_config_path'])
            with base_config_path.open('r') as base_config_fin:
                base_config = json.load(base_config_fin)
            parent_config = parent_task.data.get('config', {})
            self.data['config'] = parent_config.derive(base_config).merge(data.get('config', {}))
            del self.data.entries['base_config_path']

        if self.runnable:

            assert self.file_ids        is not None
            assert self.img_pathpattern is not None

            self.    seg_pathpattern = path / self.data.entries.get(    'seg_pathpattern', None)
            self.    adj_pathpattern = path / self.data.entries.get(    'adj_pathpattern', None)
            self.    log_pathpattern = path / self.data.entries.get(    'log_pathpattern', None)
            self.    cfg_pathpattern = path / self.data.entries.get(    'cfg_pathpattern', None)
            self.overlay_pathpattern = path / self.data.entries.get('overlay_pathpattern', None)
            self.        result_path = path / DATA_DILL_GZ_FILENAME
            self.       timings_path = path / 'timings.csv'
            self.   performance_path = path / 'performance.csv'
            self.  timings_json_path = path / '.timings.json'
            self.        digest_path = path / '.digest'
            self.    digest_cfg_path = path / '.digest.cfg.json'
            self.             config = self.data.get('config', {})
            self.         seg_border = self.data.entries.get('seg_border', None)
            self.             dilate = self.data.entries.get('dilate', 0)
            self.    merge_threshold = self.data.entries.get('merge_overlap_threshold', np.infty)
            self.         last_stage = self.data.entries.get('last_stage', None)
            self.            environ = self.data.entries.get('environ', {})

    def resolve_path(self, path):
        if path is None: return None
        path = pathlib.Path(os.path.expanduser(str(path))
            .replace('{DIRNAME}', self.path.name)
            .replace('{ROOTDIR}', str(self.root_path)))
        if path.is_absolute():
            return path.resolve()
        else:
            return path.resolve().relative_to(os.getcwd())

    @staticmethod
    def create_from_directory(task_dir, parent_task, override_cfg={}, force_runnable=False):
        """Instantiates the task from the specification in a directory (see :ref:`batch_task_spec`).

        :param task_dir: The path of the directory which contains a ``task.json`` specification file.
        :param parent_task: The parent task (or ``None`` if this a root task).
        :param override_cfg: Dictionary of task specification settings which are to be overwritten.
        :param force_runnable: If ``True``, the task will be treated as runnable, regardless of the task specification.
        """
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
            except:
                raise ValueError(f'Error processing: "{task_file}"')
        return None
    
    @property
    def root_path(self):
        """The root path of the task (see :ref:`batch_system`)."""
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
        """Hash code of the hyperparameters of this task.
        """
        return self.config.md5.hexdigest()
        
    @property
    def is_pending(self):
        """``True`` if the task needs to run, and ``False`` if the task is completed or not runnable.
        """
        return self.runnable and not (self.digest_path.exists() and self.digest_path.read_text() == self.config_digest)

    def run(self, task_info=None, dry=False, verbosity=0, force=False, one_shot=False, debug=False, report=None, pickup=True, out=None):
        out = get_output(out)
        if not self.runnable: return
        _compute_objects._DEBUG = debug
        if not force and not self.is_pending:
            out.write(f'\nSkipping task: {self._fmt_path(self.path)} {"" if task_info is None else f"({task_info})"}')
            return
        if self.last_stage is not None:
            if task_info is not None: task_info = f'{task_info}, '
            else: task_info = ''
            task_info = task_info + f'last stage: {self.last_stage}'
        out.write(Text.style(f'\nEntering task: {self._fmt_path(self.path)} {"" if task_info is None else f"({task_info})"}', Text.BLUE))
        out2 = out.derive(margin=2)
        pipeline = self._initialize()
        assert self.last_stage is None or self.last_stage == '' or not np.isinf(pipeline.find(self.last_stage)), f'unknown stage "{self.last_stage}"'
        try:
            first_stage, data = self.find_first_stage_name(pipeline, dry, pickup, out=out2)
            out3 = out2.derive(margin=2, muted = (verbosity <= -int(not dry)))
            timings = self._load_timings()
            performance = PerformanceReport()
            for file_idx, file_id in enumerate(self.file_ids):
                img_filepath = str(self.img_pathpattern) % file_id
                progress = file_idx / len(self.file_ids)
                if report is not None: report.update(self, progress)
                out3.write(Text.style(f'\n[{self._fmt_path(self.path)}] ', Text.BLUE + Text.BOLD) + Text.style(f'Processing file: {img_filepath}', Text.BOLD) + f' ({100 * progress:.0f}%)')
                kwargs = dict(    img_filepath = img_filepath,
                                  seg_filepath = _resolve_pathpattern(self.seg_pathpattern    , file_id),
                                  adj_filepath = _resolve_pathpattern(self.adj_pathpattern    , file_id),
                                  log_filepath = _resolve_pathpattern(self.log_pathpattern    , file_id),
                                  cfg_filepath = _resolve_pathpattern(self.cfg_pathpattern    , file_id),
                              overlay_filepath = _resolve_pathpattern(self.overlay_pathpattern, file_id),
                              rasterize_kwargs = dict(merge_overlap_threshold=self.merge_threshold, dilate=self.dilate),
                                    seg_border = self.seg_border,
                                    last_stage = self.last_stage,
                                           cfg = self.config.copy())
                if file_id not in data: data[file_id] = None
                if self.last_stage is not None and pipeline.find(self.last_stage) < pipeline.find('postprocess'): kwargs['seg_filepath'] = None
                data[file_id], _timings = _process_file(dry, pipeline, data[file_id], first_stage=first_stage, out=out3, **kwargs)
                if not dry: _compress_logs(kwargs['log_filepath'])
                if file_id not in timings: timings[file_id] = {}
                timings[file_id].update(_timings)
                if not dry and 'performance' in data[file_id]:
                    performance += data[file_id]['performance']
            out2.write('')
            if report is not None: report.update(self, 'active')
            if not dry and not np.isnan(performance.nontrivial_pruning_success):
                out2.write(Text.style('Non-trivial pruning: ', Text.BOLD) + f'{100 * performance.nontrivial_pruning_success:.1f}% (computed {performance.nontrivial_computed_object_count} / {performance.nontrivial_object_count})')
            
            skip_writing_results_conditions = [
                one_shot,
                self.last_stage is not None and pipeline.find(self.last_stage) <= pipeline.find('dsm') and not self.result_path.exists(),
                first_stage is not None and pipeline.find(first_stage) >= pipeline.find('postprocess')
            ]
            if any(skip_writing_results_conditions):
                out2.write('Skipping writing results')
            else:
                if not dry:
                    self.write_timings(timings)
                    out2.intermediate(f'Writing results... {self._fmt_path(self.result_path)}')
                    with gzip.open(self.result_path, 'wb') as fout:
                        dill.dump(data, fout, byref=True)
                    with self.digest_cfg_path.open('w') as fout:
                        self.config.dump_json(fout)
                    _write_performance_report(self.path, self.performance_path, data, performance)
                out2.write(Text.style('Results written to: ', Text.BOLD) + self._fmt_path(self.result_path))
            if not dry and not one_shot: self.digest_path.write_text(self.config_digest)
            for obj_name in ('data', 'shallow_data'):
                if obj_name in locals(): return locals()[obj_name]
        except:
            out.write(Text.style(f'\nError while processing task: {self._fmt_path(self.path)}', Text.RED))
            raise
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
            first_stage = find_first_differing_stage(pipeline, self.config.entries, previous_task.config.entries)
            pickup_candidates.append((previous_task, first_stage))
        if self.result_path.exists() and self.digest_cfg_path.exists():
            with self.digest_cfg_path.open('r') as fin:
                config = json.load(fin)
            first_stage = find_first_differing_stage(pipeline, self.config.entries, config)
            pickup_candidates.append((self, first_stage))
        return pickup_candidates

    def find_best_pickup_candidate(self, pipeline):
        pickup_candidates = self.find_pickup_candidates(pipeline)
        if len(pickup_candidates) == 0: return None, None
        pickup_candidate_scores = [pipeline.find(first_stage) for task, first_stage in pickup_candidates]
        return pickup_candidates[np.argmax(pickup_candidate_scores)]

    def find_first_stage_name(self, pipeline, dry=False, pickup=True, out=None):
        out = get_output(out)
        pickup_task, stage_name = self.find_best_pickup_candidate(pipeline) if pickup else (None, None)
        if pickup_task is None or pipeline.find(stage_name) <= pipeline.find('dsm') + 1:
            return None, {}
        else:
            out.write(f'Picking up from: {self._fmt_path(pickup_task.result_path)} ({stage_name if stage_name != "" else "load"})')
            if not dry:
                with gzip.open(pickup_task.result_path, 'rb') as fin:
                    data = dill.load(fin)
                return stage_name, data
            else:
                return stage_name, {}

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
    """Loads all tasks from a given directory (see :ref:`batch_task_spec`).

    :param override_cfg: Dictionary of task specification settings which are to be overwritten.
    """

    def __init__(self, override_cfg={}):
        self.tasks        = []
        self.override_cfg = override_cfg

    def load(self, path):
        """Loads all task from the root directory ``path``.
        """
        root_path = pathlib.Path(path)
        self._process_directory(root_path)

    def _process_directory(self, current_dir, parent_task=None):
        task = Task.create_from_directory(current_dir, parent_task, self.override_cfg)
        if task is not None:
            self.tasks.append(task)
            parent_task = task
        for d in os.listdir(current_dir):
            f = current_dir / d
            if f.is_dir():
                self._process_directory(f, parent_task)


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
    parser.add_argument('--fresh', help='do not pick up previous results', action='store_true')
    parser.add_argument('--task', help='run only the given task', type=str, default=[], action='append')
    parser.add_argument('--task-dir', help='run only the given task and those from its sub-directories', type=str, default=[], action='append')
    parser.add_argument('--debug', help='do not use multiprocessing', action='store_true')
    parser.add_argument('--report', help='report current status to file', type=str, default='/tmp/superdsm-status')
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
            try:
                task.run(task_info, dry, args.verbosity, args.force, args.oneshot, args.debug, report, not args.fresh, out)
            except:
                report.update(task, 'error')
                raise
            os._exit(0)
        else:
            if os.waitpid(newpid, 0)[1] != 0:
                out.write('An error occurred: interrupting')
                sys.exit(1)
            else:
                report.update(task, 'done')
    out.write(f'\nRan {run_task_count} task(s) out of {len(runnable_tasks)} in total')

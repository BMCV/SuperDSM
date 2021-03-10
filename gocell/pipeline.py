import gocell.aux
import gocell.config
import gocell.surface

import math
import numpy as np
import time


class Stage(object):

    def __init__(self, name, cfg_key=None, inputs=[], outputs=[]):
        if cfg_key is None: cfg_key = name
        self.name    = name
        self.cfg_key = cfg_key
        self.inputs  = dict([(key, key) for key in  inputs])
        self.outputs = dict([(key, key) for key in outputs])
        self._callbacks = {}

    def _callback(self, name, *args, **kwargs):
        if name in self._callbacks:
            for cb in self._callbacks[name]:
                cb(name, *args, **kwargs)

    def add_callback(self, name, cb):
        if name not in self._callbacks: self._callbacks[name] = []
        self._callbacks[name].append(cb)

    def remove_callback(self, name, cb):
        if name in self._callbacks: self._callbacks[name].remove(cb)

    def __call__(self, data, cfg, out=None, log_root_dir=None):
        out = gocell.aux.get_output(out)
        cfg = gocell.config.get_value(cfg, self.cfg_key, {})
        if gocell.config.get_value(cfg, 'enabled', self.ENABLED_BY_DEFAULT):
            out.intermediate(f'Starting stage "{self.name}"')
            self._callback('start', data)
            input_data = {}
            for data_key, input_data_key in self.inputs.items():
                input_data[input_data_key] = data[data_key]
            t0 = time.time()
            output_data = self.process(input_data, cfg=cfg, out=out, log_root_dir=log_root_dir)
            dt = time.time() - t0
            assert len(set(output_data.keys()) ^ set(self.outputs)) == 0, 'stage "%s" generated unexpected output' % self.name
            for output_data_key, data_key in self.outputs.items():
                data[data_key] = output_data[output_data_key]
            self._callback('end', data)
            return dt
        else:
            out.write(f'Skipping disabled stage "{self.name}"')
            self._callback('skip', data)
            return 0

    def process(self, input_data, cfg, out, log_root_dir):
        raise ValueError('not implemented')


class ProcessingControl:

    def __init__(self, first_stage=None, last_stage=None):
        self.started     = True if first_stage is None else False
        self.first_stage = first_stage
        self.last_stage  =  last_stage
    
    def step(self, stage):
        if not self.started and stage == self.first_stage: self.started = True
        do_step = self.started
        if stage == self.last_stage: self.started = False
        return do_step


class Pipeline:

    def __init__(self):
        self.stages = []

    def process_image(self, g_raw, cfg, first_stage=None, last_stage=None, data=None, out=None, log_root_dir=None):
        assert 'preprocess1' not in cfg.keys(), 'config version is deprecated'
        cfg = gocell.aux.copy_dict(cfg)
        if log_root_dir is not None: gocell.aux.mkdir(log_root_dir)
        if first_stage == self.stages[0].name and data is None: first_stage = None
        if first_stage is not None and first_stage.endswith('+'): first_stage = self.stages[1 + self.find(first_stage[:-1])].name
        if first_stage is not None and last_stage is not None and self.find(first_stage) > self.find(last_stage): return data, cfg, {}
        out  = gocell.aux.get_output(out)
        ctrl = ProcessingControl(first_stage, last_stage)
        if ctrl.step('init'): data = self.init(g_raw, cfg)
        else: assert data is not None, 'data argument must be provided if first_stage is used'
        timings = {}
        for stage in self.stages:
            if ctrl.step(stage.name):
                dt = stage(data, cfg, out=out, log_root_dir=log_root_dir)
                timings[stage.name] = dt
        return data, cfg, timings

    def init(self, g_raw, cfg):
        if gocell.config.get_value(cfg, 'histological', False):
            g_rgb = g_raw
            g_raw = g_raw.mean(axis=2)
            g_raw = g_raw.max() - g_raw
        else:
            g_rgb = None
        data = dict(g_raw = gocell.surface.Surface.create_from_image(g_raw).model) ## does some normalization
        if g_rgb is not None:
            data['g_rgb'] = g_rgb
        return data

    def find(self, stage_name, not_found_dummy=np.inf):
        try:
            return [stage.name for stage in self.stages].index(stage_name)
        except ValueError:
            return not_found_dummy

    def append(self, stage, after=None):
        if after is None: self.stages.append(stage)
        else:
            if isinstance(after, str): after = self.find(after)
            self.stages.insert(after + 1, stage)


def create_pipeline(stages):
    available_inputs = set(['g_raw'])
    remaining_stages = list(stages)

    pipeline = Pipeline()
    while len(remaining_stages) > 0:
        next_stage = None
        for stage in remaining_stages:
            if frozenset(stage.inputs.keys()).issubset(available_inputs):
                next_stage = stage
                break
        if next_stage is None:
            raise ValueError('failed to resolve total ordering')
        remaining_stages.remove(next_stage)
        pipeline.append(next_stage)
        available_inputs |= frozenset(next_stage.outputs.keys())

    return pipeline


def create_default_pipeline():
    import gocell.preprocessing
    import gocell.modelfit_config
    import gocell.topdownsegm
    import gocell.generations
    import gocell.postprocessing

    stages = [
        gocell.preprocessing.PreprocessingStage(),
        gocell.modelfit_config.ModelfitConfigStage(),
        gocell.topdownsegm.TopDownSegmentation(),
        gocell.generations.GenerationStage(),
        gocell.postprocessing.Postprocessing(),
    ]

    return create_pipeline(stages)


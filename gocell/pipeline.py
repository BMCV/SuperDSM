import gocell.aux
import gocell.config
import gocell.surface

import math
import numpy as np


class Stage(object):

    def __init__(self, name, cfg_key=None, inputs=[], outputs=[]):
        if cfg_key is None: cfg_key = name
        self.name    = name
        self.cfg_key = cfg_key
        self.inputs  = dict([(key, key) for key in  inputs])
        self.outputs = dict([(key, key) for key in outputs])

    def __call__(self, data, cfg, out=None, log_root_dir=None):
        out = gocell.aux.get_output(out)
        input_data = {}
        for data_key, input_data_key in self.inputs.items():
            input_data[input_data_key] = data[data_key]
        output_data = self.process(input_data, cfg=gocell.config.get_value(cfg, self.cfg_key, {}), out=out, log_root_dir=log_root_dir)
        assert len(set(output_data.keys()) ^ set(self.outputs)) == 0, 'stage "%s" generated unexpected output' % self.name
        for output_data_key, data_key in self.outputs.items():
            data[data_key] = output_data[output_data_key]

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
        if log_root_dir is not None: gocell.aux.mkdir(log_root_dir)
        if first_stage == self.stages[0].name and data is None: first_stage = None
        out  = gocell.aux.get_output(out)
        ctrl = ProcessingControl(first_stage, last_stage)
        if ctrl.step('init'): data = self.init(g_raw, cfg)
        else: assert data is not None, 'data argument must be provided if first_stage is used'
        for stage in self.stages:
            if ctrl.step(stage.name): stage(data, cfg, out=out, log_root_dir=log_root_dir)
        return data, cfg

    def init(self, g_raw, cfg):
        return {
            'g_raw': gocell.surface.Surface.create_from_image(g_raw).model  ## does some normalization
        }

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
    import gocell.seeds
    import gocell.atoms
    import gocell.generations

    stages = [
        gocell.preprocessing.PreprocessingStage1(),
        gocell.preprocessing.PreprocessingStage2(),
        gocell.seeds.SeedStage(),
        gocell.atoms.AtomStage(),
        gocell.generations.GenerationStage()
    ]

    return create_pipeline(stages)


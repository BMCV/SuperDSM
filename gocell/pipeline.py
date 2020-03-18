import gocell.aux      as aux
import gocell.config   as config
import gocell.surface  as surface
import gocell.modelfit as modelfit
import math


class Stage(object):

    def __init__(self, name, cfg_key=None, inputs=[], outputs=[]):
        if cfg_key is None: cfg_key = name
        self.name    = name
        self.cfg_key = cfg_key
        self.inputs  = dict([(key, key) for key in  inputs])
        self.outputs = dict([(key, key) for key in outputs])

    def __call__(self, data, cfg, out=None):
        out = aux.Output.get(out)
        input_data = {}
        for data_key, input_data_key in self.inputs.items():
            input_data[input_data_key] = data[data_key]
        output_data = self.process(input_data, cfg=config.get_value(cfg, self.cfg_key, {}), out=out)
        assert len(set(output_data.keys()) ^ set(self.outputs)) == 0, 'stage "%s" generated unexpected output' % self.name
        for output_data_key, data_key in self.outputs.items():
            data[data_key] = output_data[output_data_key]

    def process(self, input_data, cfg, out):
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

    def process_image(self, g_raw, cfg, first_stage=None, last_stage=None, data=None, out=None):
        out  = aux.Output.get(out)
        ctrl = ProcessingControl(first_stage, last_stage)
        if ctrl.step('init'): data = self.init(g_raw, cfg)
        else: assert data is not None, 'data argument must be provided if first_stage is used'
        for stage in self.stages:
            if ctrl.step(stage.name): stage(data, cfg, out=out)
        return data, cfg

    def init(self, g_raw, cfg):
        return {
            'g_raw': surface.Surface.create_from_image(g_raw).model,  ## does some normalization
            'min_region_size': 2 * math.pi * config.get_value(cfg, 'min_region_radius', 25)
        }

    def find(self, stage_name):
        return [stage.name for stage in self.stages].index(stage_name)

    def append(self, stage, after=None):
        if after is None: self.stages.append(stage)
        else:
            if isinstance(after, str): after = self.find(after)
            self.stages.insert(after + 1, stage)


def create_default_pipeline(backend, log_seeds=False, selection_type='maxsetpack'):
    from gocell.preprocessing  import Preprocessing
    from gocell.superpixels    import Seeds, GaussianLaplaceSeeds, Superpixels, SuperpixelsEntropy, SuperpixelsDiscard
    from gocell.candidates     import ComputeCandidates, FilterUniqueCandidates, IntensityModels, ProcessCandidates, AnalyzeCandidates
    from gocell.maxsetpack     import MaxSetPackWeights, MaxSetPackGreedy, MaxSetPackCheck
    from gocell.minsetcover    import MinSetCoverWeights, MinSetCoverGreedy, MinSetCoverCheck
    from gocell.postprocessing import Postprocessing

    if isinstance(backend, int): backend = modelfit.fork_based_backend(num_forks=backend)

    pipeline = Pipeline()

    pipeline.append(Preprocessing())
    pipeline.append(GaussianLaplaceSeeds() if log_seeds else Seeds())
    pipeline.append(Superpixels())
    if not log_seeds:
        pipeline.append(SuperpixelsEntropy())
        pipeline.append(SuperpixelsDiscard())
    pipeline.append(ComputeCandidates())
    pipeline.append(FilterUniqueCandidates())
    pipeline.append(IntensityModels())
    pipeline.append(ProcessCandidates(backend))
    pipeline.append(AnalyzeCandidates())
    if selection_type == 'maxsetpack':
        pipeline.append(MaxSetPackWeights())
        pipeline.append(MaxSetPackGreedy())
        pipeline.append(MaxSetPackCheck())
    elif selection_type == 'minsetcover':
        pipeline.append(MinSetCoverWeights())
        pipeline.append(MinSetCoverGreedy())
        pipeline.append(MinSetCoverCheck())
    else:
        raise ValueError('unknown selection_type "%s"' % selection_type)
    pipeline.append(Postprocessing())

    return pipeline


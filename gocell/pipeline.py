import aux
import config


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
        output_data = self.process(input_data, config.get_value(cfg, self.cfg_key, {}))
        assert len(set(output_data.keys()) ^ set(self.outputs)) == 0, 'stage "%s" generated unexpected output' % self.name
        for output_data_key, data_key in self.outputs.items():
            data[data_key] = output_data[output_data_key]

    def process(self, input_data, cfg, out):
        raise ValueError('not implemented')


class ProcessingControl:

    def __init__(self, first_stage=None):
        self.started     = True if first_stage is None else False
        self.first_stage = first_stage
    
    def step(self, stage):
        if not self.started and stage == self.first_stage: self.started = True
        return self.started


class Pipeline:

    def __init__(self):
        self.stages = []

    def process_image(g_raw, cfg, first_stage=None, data=None, out=None):
        out  = aux.Output.get(out)
        ctrl = ProcessingControl(first_stage)
        if ctrl.step('init'): data = self.init(g_raw, cfg)
        else: assert data is not None, 'data argument must be provided if first_stage is used'
        for stage in self.stages:
            if ctrl.step(stage.name): stage(data, cfg, out=out)
        return data

    def init(self, g_raw, cfg):
        return {
            'g_raw': g_raw,
            'min_candidate_size': 2 * pi * config.get_value(cfg, 'min_candidate_radius', 25)
        }

    def find(self, stage_name):
        return [stage.name for stage in self.stages].index(stage_name)

    def append(self, stage, after=None):
        if after is None: self.stages.append(stage)
        else:
            if isinstance(after, str): after = self.find(after)
            self.stages.insert(after + 1, stage)


def create_default_pipeline(backend):
    from preprocessing  import Preprocessing
    from superpixels    import Seeds, Superpixels, SuperpixelsEntropy, SuperpixelsDiscard
    from candidates     import ComputeCandidates, FilterUniqueCandidates, ProcessCandidates, AnalyzeCandidates
    from maxsetpack     import MaxSetPackWeights, MaxSetPackGreedy
    from postprocessing import Postprocessing

    if isinstance(backend, (int, long)): backend = modelfit.fork_based_backend(num_forks=backend)

    pipeline = Pipeline()

    pipeline.append(Preprocessing())
    pipeline.append(Seeds())
    pipeline.append(Superpixels())
    pipeline.append(SuperpixelsEntropy())
    pipeline.append(SuperpixelsDiscard())
    pipeline.append(ComputeCandidates())
    pipeline.append(FilterUniqueCandidates())
    pipeline.append(ProcessCandidates(backend))
    pipeline.append(AnalyzeCandidates())
    pipeline.append(MaxSetPackWeights())
    pipeline.append(MaxSetPackGreedy())
    pipeline.append(Postprocessing())

    return pipeline


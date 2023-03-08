from .pipeline import Stage


MODELFIT_KWARGS_DEFAULTS = {
    'cachesize': 1,
    'sparsity_tol': 0,
    'init': 'elliptical',
    'smooth_amount': 10,
    'epsilon': 1.0,
    'alpha': 0.5,
    'scale': 1000,
    'smooth_subsample': 20,
    'gaussian_shape_multiplier': 2,
    'smooth_mat_dtype': 'float32',
    'min_background_margin': 20,
    'cp_timeout': 300
}


class ModelfitConfigStage(Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(ModelfitConfigStage, self).__init__('modelfit', inputs=[], outputs=['mfcfg'])

    def process(self, input_data, cfg, out, log_root_dir):
        mfcfg = {
            key: cfg.get(key, MODELFIT_KWARGS_DEFAULTS[key]) for key in MODELFIT_KWARGS_DEFAULTS.keys()
        }
        
        return {
            'mfcfg': mfcfg
        }


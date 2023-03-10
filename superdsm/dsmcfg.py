from .pipeline import Stage


DSM_CONFIG_DEFAULTS = {
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


class DSM_ConfigStage(Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(DSM_ConfigStage, self).__init__('dsm', inputs=[], outputs=['dsm_cfg'])

    def process(self, input_data, cfg, out, log_root_dir):
        dsm_cfg = {
            key: cfg.get(key, DSM_CONFIG_DEFAULTS[key]) for key in DSM_CONFIG_DEFAULTS.keys()
        }
        
        return {
            'dsm_cfg': dsm_cfg
        }


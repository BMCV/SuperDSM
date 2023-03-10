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
    """Fetches the hyperparameters from the ``dsm`` namespace and provides them as an output.

    The purpose of this stage is to provide the hyperparameters from the ``dsm`` namespace as the output ``dsm_cfg``, which can be used by any other stage. This concept enables any stage, like :py:class:`~.c2freganal.C2F_RegionAnalysis` and :py:class:`~.globalenergymin.GlobalEnergyMinimization`, to access the DSM-related hyperparameters without having to access the ``dsm`` hyperparameter namespace. Refer to :ref:`pipeline_inputs_and_outputs` for more information on the available inputs and outputs.

    Hyperparameters
    ---------------

    The following hyperparameters are fetched:

    ``dsm/cachesize``
        tbd.

    ``dsm/sparsity_tol``
        tbd.

    ``dsm/init``
        tbd.

    ``dsm/smooth_amount``
        tbd.

    ``dsm/epsilon``
        tbd.

    ``dsm/alpha``
        tbd.

    ``dsm/scale``
        tbd.

    ``dsm/smooth_subsample``
        tbd.

    ``dsm/gaussian_shape_multiplier``
        tbd.

    ``dsm/smooth_mat_dtype``
        tbd.

    ``dsm/min_background_margin``
        tbd.

    ``dsm/cp_timeout``
        tbd.
    """

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


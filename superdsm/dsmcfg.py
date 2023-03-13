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

    The purpose of this stage is to provide the hyperparameters from the ``dsm`` namespace as the output ``dsm_cfg``, which can be used by any other stage.
    This concept enables any stage to access the DSM-related hyperparameters, like the :py:class:`~.c2freganal.C2F_RegionAnalysis` and :py:class:`~.globalenergymin.GlobalEnergyMinimization` stages, without having to access the ``dsm`` hyperparameter namespace.
    Refer to :ref:`pipeline_inputs_and_outputs` for more information on the available inputs and outputs.

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
        Corresponds to :math:`\sigma_G` described in :ref:`pipeline_theory_dsm`. Defaults to 10, or to ``AF_smooth_amount × scale`` if computed automatically (forced to :math:`\geq 4` and ``AF_smooth_amount`` defaults to 0.2).

    ``dsm/smooth_subsample``
        Corresponds to the amount of sub-sampling used to obtain the matrix :math:`\tilde G_\omega` in the :ref:`paper <references>` (Section 3.3). Defaults to 20, or to ``AF_smooth_subsample × scale`` if computed automatically (forced to :math:`\geq 8` and ``AF_smooth_subsample`` defaults to 0.4).

    ``dsm/epsilon``
        tbd.

    ``dsm/alpha``
        Governs the regularization of the deformations and corresponds to :math:`\\alpha` described in :ref:`pipeline_theory_cvxprog`. Increasing this value leads to a smoother segmentation result. Defaults to 0.5, or to ``AF_alpha × scale^2`` if computed automatically (where ``AF_alpha`` corresponds to :math:`\alpha_\text{factor}` in the :ref:`paper <references>` and defaults to 5e-4).

    ``dsm/scale``
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

    def configure(self, scale, radius, diameter):
        return {
            'alpha': (scale ** 2, 0.0005),
            'smooth_amount': (scale, 0.2, dict(type=int, min=4)),
            'smooth_subsample':  (scale, 0.4, dict(type=int, min=8)),
        }


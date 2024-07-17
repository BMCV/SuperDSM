from .pipeline import Stage

import math
import scipy.ndimage as ndi
import skimage.morphology as morph
import numpy as np


class Preprocessing(Stage):
    """Implements the computation of the intensity offsets and the edge map (see :ref:`pipeline_theory_cvxprog`).

    This stage requires ``g_raw`` for input (the input image) and produces ``y`` and ``z`` for output (the offset image intensities and the edge map). Refer to :ref:`pipeline_inputs_and_outputs` for more information on the available inputs and outputs.

    Hyperparameters
    ---------------

    The following hyperparameters can be used to control this pipeline stage:

    ``preprocess/sigma1``
        The scale of the Gaussian filter used for denoising. Defaults to :math:`\\sqrt{2}`.

    ``preprocess/sigma2``
        The scale of the Gaussian filter :math:`\\mathcal G_\\sigma`, which is used to determine the intensity offsets :math:`\\tau_x` (see :ref:`pipeline_theory_cvxprog`). Defaults to :math:`40`, or to ``AF_sigma2 × scale`` if configured automatically (and ``AF_sigma2`` defaults to 1).

    ``preprocess/sigma3``
        The scale of the Gaussian filter used for computation of the edge map. Defaults to :math:`4`, or to ``AF_sigma3 × scale`` if configured automatically (and ``AF_sigma3`` defaults to 0.1).

    ``preprocess/offset_clip``
        Corresponds to :math:`\\tau_\\text{max}` in Supplemental Material 1. Defaults to :math:`3`.

    ``preprocess/lower_clip_mean``
        If ``True``, intensity offsets :math:`\\tau_x` smaller than the mean image intensity are set to the mean image intensity. Defaults to ``False``.
    """

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(Preprocessing, self).__init__('preprocess',
                                            inputs  = ['g_raw'],
                                            outputs = ['y', 'z'])

    def process(self, input_data, cfg, out, log_root_dir):
        g_raw = input_data['g_raw']

        sigma1 = cfg.get('sigma1', math.sqrt(2))
        sigma2 = cfg.get('sigma2', 40)
        sigma3 = cfg.get('sigma3', 4)
        offset_clip  = cfg.get('offset_clip', 3)
        lower_clip_mean = cfg.get('lower_clip_mean', False)

        offset_original = ndi.gaussian_filter(g_raw, sigma2)
        if np.isinf(offset_clip):
            offset_combined = offset_original

        else:
            offset_clip_abs = offset_clip * g_raw.std()
            offset_clipped  = ndi.gaussian_filter(g_raw.clip(0, offset_clip_abs), sigma2)

            clip_area = (g_raw > offset_clip_abs)
            _tmp1 = ndi.distance_transform_edt(~clip_area)
            _tmp1 = (sigma2 - _tmp1).clip(0, np.inf)
            _tmp1 = (_tmp1 / _tmp1.max()) ** 2
            offset_combined = (1 - _tmp1) * offset_clipped + _tmp1 * offset_original
            
        if lower_clip_mean:
            offset_combined = np.max([offset_combined, np.full(g_raw.shape, g_raw.mean())], axis=0)

        y = ndi.gaussian_filter(g_raw, sigma1) - offset_combined
        z = ndi.gaussian_gradient_magnitude(g_raw, sigma3)
        z[y < 0] = 0
        z -= z.mean()
        
        return {
            'y': y,
            'z': z,
        }

    def configure_ex(self, scale, radius, diameter):
        return {
            'sigma2': (scale, 1.0),
            'sigma3': (scale, 0.1),
        }


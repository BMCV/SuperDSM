.. _hyperparameters:

Hyperparameters
===============

Each pipeline stage can be controlled by a separate set of hyperparameters. Refer to the documentation of the pipeline stages for details.

Automation
----------

In addition, several hyperparameters can be set automatically based on the scale :math:`\sigma` of objects in an image. The scale of the objects is estimated automatically as described in Section 3.1 of the paper (:ref:`Kostrykin and Rohr, 2023 <references>`). The current implementation determines values corresponding to object radii between 20 and 200 pixels. If the hyperparameter ``AF_sigma`` is set, then the scale :math:`\sigma` is forced to its value and the automatic scale detection is skipped. The hyperparameter ``AF_sigma`` is not set by default.

Hyperparameters are only set automatically based on the scale of objects, when the :py:mod:`~superdsm.automation` module (as in :ref:`this <usage_example_interactive>` example) or batch processing are used (as in :ref:`this <usage_example_batch>` example). Hyperparameters are *not* set automatically if py:meth:`~superdsm.pipeline.Pipeline.process_image` is used directly.

The following hyperparameters can be set automatically based on the scale of objects. In the formulas given below, ``scale`` corresponds to :math:`\sigma`, ``radius`` corresponds to :math:`\sqrt{2} \cdot \sigma`, and ``diameter`` corresponds to :math:`\sqrt{8} \cdot \sigma`.

``preprocess/sigma2``
^^^^^^^^^^^^^^^^^^^^^

Stage: :py:class:`~superdsm.preprocessing.Preprocessing`

TBC


``global-energy-minimization/beta``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage: :py:class:`~superdsm.globalenergymin.GlobalEnergyMinimization`

Corresponds to the constant term :math:`\beta` of the extended set energy functions in the paper (:ref:`Kostrykin and Rohr, 2023 <references>`, Sections 2.3.2 and 3.3).

Defaults to ``AF_beta × scale^2`` (where ``AF_beta`` corresponds to :math:`\beta_\text{factor}` in the paper and defaults to 0.66). Due to a transmission error, the values reported for ``AF_beta`` in the paper were misstated by a factor of 2 (Section 3.3, Supplemental Material 8).

``global-energy-minimization/max_seed_distance``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage: :py:class:`~superdsm.globalenergymin.GlobalEnergyMinimization`

Maximum distance allowed between two seed points of atomic image regions which are grouped into an image region corresponding to single object. This can be used to enforce that the segmented objects will be of a maximum size, and thus to limit the computational cost by using prior knowledge.

Defaults to ``AF_max_seed_distance × diameter`` (and ``AF_max_seed_distance`` defaults to infinity).

``postprocess/min_object_radius``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage: :py:class:`~superdsm.postprocessing.Postprocessing`

Objects smaller than a circle of this radius are discarded.

Defaults to ``AF_min_object_radius × radius`` (and ``AF_min_object_radius`` defaults to zero).

``postprocess/max_object_radius``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage: :py:class:`~superdsm.postprocessing.Postprocessing`

Objects larger than a circle of this radius are discarded.

Defaults to ``AF_max_object_radius × radius`` (and ``AF_max_object_radius`` defaults to infinity).

``postprocess/min_glare_radius``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage: :py:class:`~superdsm.postprocessing.Postprocessing`

Corresponds to the minimum object radius required for an object to be possibly recognized as an autofluorescence artifact.

Defaults to ``AF_min_glare_radius × radius`` (and ``AF_min_glare_radius defaults`` to infinity).

``modelfit/alpha``
^^^^^^^^^^^^^^^^^^

Stage: :py:class:`~superdsm.modelfit_config.ModelfitConfigStage`

Governs the regularization of the deformations and corresponds to :math:`\alpha` in the paper (:ref:`Kostrykin and Rohr, 2023 <references>`, Sections 2.2 and 3.3). Increasing this value leads to a smoother segmentation result.

Defaults to ``AF_alpha × scale^2`` (where ``AF_alpha`` corresponds to :math:`\alpha_\text{factor}` in the paper and defaults to 5e-4).

``modelfit/smooth_amount``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage: :py:class:`~superdsm.modelfit_config.ModelfitConfigStage`

Corresponds to :math:`\sigma_G` in the paper (:ref:`Kostrykin and Rohr, 2023 <references>`, Section 3.3).

Defaults to ``AF_smooth_amount × scale`` (forced to :math:`\geq 4` and ``AF_smooth_amount`` defaults to 0.2).

``modelfit/smooth_subsample``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage: :py:class:`~superdsm.modelfit_config.ModelfitConfigStage`

Corresponds to the amount of sub-sampling used to obtain the matrix :math:`\tilde G_\omega` in the paper (:ref:`Kostrykin and Rohr, 2023 <references>`, Section 3.3).

Defaults to ``AF_smooth_subsample × scale`` (forced to :math:`\geq 8` and ``AF_smooth_subsample`` defaults to 0.4).

``c2f-region-analysis/min_region_radius``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage: :py:class:`~superdsm.c2freganal.C2F_RegionAnalysis`

No region determined by the coarse-to-fine region analysis scheme is smaller than a circle of this radius (see Section 3.2 and Supplemental Material 5 in :ref:`Kostrykin and Rohr, 2023 <references>`).

Defaults to ``AF_min_region_radius × radius`` (and ``AF_min_region_radius`` defaults to 0.33).


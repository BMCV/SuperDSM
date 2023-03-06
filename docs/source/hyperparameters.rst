.. _hyperparameters:

Hyperparameters
===============

Each pipeline stage can be controlled by a separate set of hyperparameters. Refer to the documentation of the pipeline stages for details.

In addition, several hyperparameters can be set automatically based on the scale :math:`\sigma` of objects in an image. The scale of the objects is estimated automatically as described in Section 3.1 of the paper (:ref:`Kostrykin and Rohr, 2023 <references>`). The current implementation determines values corresponding to object radii between 20 and 200 pixels. If the hyperparameter ``AF_sigma`` is set, then the scale :math:`\sigma` is forced to its value and the automatic scale detection is skipped. The hyperparameter ``AF_sigma`` is not set by default.

The following hyperparameters can be set automatically based on the scale of objects. In the formulas given below, ``scale`` corresponds to :math:`\sigma`, ``radius`` corresponds to :math:`\sqrt{2} \cdot \sigma`, and ``diameter`` corresponds to :math:`\sqrt{8} \cdot \sigma`.

``preprocess/sigma2``
---------------------

Stage: :py:class:`~superdsm.preprocessing.Preprocessing`

TBC


``generations/alpha``
---------------------

Stage: :py:class:`~superdsm.generations.GenerationStage`

Corresponds to :math:`\beta` in the paper (Section 2.3.2 and Section 3.3).

Defaults to ``AF_beta × scale^2`` (and ``AF_beta`` defaults to 0.66).

``generations/max_seed_distance``
---------------------------------

Stage: :py:class:`~superdsm.generations.GenerationStage`

TBC

``postprocess/min_obj_radius``
------------------------------

Stage: :py:class:`~superdsm.postprocessing.Postprocessing`

Corresponds to ``min_object_radius`` in the paper.

Defaults to ``AF_min_object_radius × radius`` (and ``AF_min_object_radius`` defaults to zero).

``postprocess/max_obj_radius``
------------------------------

Stage: :py:class:`~superdsm.postprocessing.Postprocessing`

Objects larger than a circle of this radius are discarded.

Defaults to ``AF_max_obj_radius × radius`` (and ``AF_max_obj_radius`` defaults to infinity).

``postprocess/min_glare_radius``
--------------------------------

Stage: :py:class:`~superdsm.postprocessing.Postprocessing`

Corresponds to minimum object radius required for an object to be possibly recognized as an autofluorescence artifact.

Defaults to ``AF_min_glare_radius × radius`` (and ``AF_min_glare_radius defaults`` to infinity).

``modelfit/rho``
----------------

Stage: :py:class:`~superdsm.modelfit_config.ModelfitConfigStage`

Corresponds to :math:`\alpha` in the paper (Section 2.2 and Section 3.3).

Defaults to ``AF_rho × scale^2`` (and ``AF_rho defaults`` to 5e-4).

``modelfit/smooth_amount``
--------------------------

Stage: :py:class:`~superdsm.modelfit_config.ModelfitConfigStage`

Corresponds to :math:`\sigma_G` in the paper (Section 3.3).

Defaults to ``AF_smooth_amount × scale`` (forced to :math:`\geq 4` and ``AF_smooth_amount`` defaults to 0.2).

``modelfit/smooth_subsample``
-----------------------------

Stage: :py:class:`~superdsm.modelfit_config.ModelfitConfigStage`

Corresponds to the amount of sub-sampling used to obtain the matrix :math:`\tilde G_\omega` in the paper (Section 3.3).

Defaults to ``AF_smooth_subsample × scale`` (forced to :math:`\geq 8` and ``AF_smooth_subsample defaults`` defaults to 0.4).

``top-down-segmentation/min_region_radius``
-------------------------------------------

Stage: :py:class:`~superdsm.topdownsegm.TopDownSegmentation`

Corresponds to "min_region_radius" in the paper (coarse-to-fine region analysis, Section 3.2).

Defaults to ``AF_min_region_radius × radius`` (and ``AF_min_region_radius defaults`` to 0.33).

``top-down-modelfit/min_background_margin``
-------------------------------------------

Stage: :py:class:`~superdsm.topdownsegm.TopDownSegmentation`

TBC


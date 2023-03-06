.. _hyperparameters:

Hyperparameters
===============

Each pipeline stage can be controlled by a separate set of hyperparameters. Refer to the documentation of the pipeline stages for details.

In addition, several hyperparameters can be set automatically based on the scale :math:`\sigma` of objects in an image. The scale of the objects is estimated automatically as described in Section 3.1 of the paper (:ref:`Kostrykin and Rohr, 2023 <references>`). The current implementation determines values corresponding to radii between 20 and 200 pixels. If the hyperparameter ``AF_sigma`` is set, then the scale :math:`\sigma` is forced to its value and the automatic scale detection is skipped. The hyperparameter ``AF_sigma`` is not set by default.

The following hyperparameters can be set automatically based on the scale :math:`\sigma`:

* ``preprocess/sigma2``

* ``generations/alpha`` Corresponds to \beta in the paper (Section 2.3.2 and Section 3.3). Defaults to AF_beta × radius^2 (AF_beta defaults to 0.33). TODO: This should be changed to AF_beta × scale^2 to be the same as in the paper (where AF_beta would default to 0,66 instead of 0,33).

* ``generations/max_seed_distance``

* ``postprocess/min_obj_radius`` Corresponds to "min_object_radius" in the paper. Defaults to AF_min_object_radius × radius (AF_min_object_radius defaults to zero).

* ``postprocess/max_obj_radius`` Objects with radius larger than that are discarded. Defaults to AF_max_obj_radius × radius (AF_max_obj_radius defaults to infinity).

* ``postprocess/min_glare_radius`` Corresponds to minimum object radius required for the object to be recognized as an autofluorescence artifact. Defaults to AF_min_glare_radius × radius (AF_min_glare_radius defaults to infinity).

* ``modelfit/rho`` Corresponds to \alpha in the paper (Section 2.2 and Section 3.3). Defaults to AF_rho × scale^2 (AF_rho defaults to 5 × 10^-4).

* ``modelfit/smooth_amount`` Corresponds to \sigma_G in the paper (Section 3.3). Defaults to AF_smooth_amount × scale (forced to ≥4, AF_smooth_amount defaults to 0,2).

* ``modelfit/smooth_subsample`` Corresponds to the amount of sub-sampling used to obtain the matrix \tilde G_\omega in the paper (Section 3.3). Defaults to AF_smooth_subsample × scale (forced to ≥8, AF_smooth_subsample defaults to 0,4).

* ``top-down-segmentation/min_region_radius`` Corresponds to "min_region_radius" in the paper (coarse-to-fine region analysis, Section 3.2). Defaults to AF_min_region_radius × radius (AF_min_region_radius defaults to 0,33).

* ``top-down-modelfit/min_background_margin``

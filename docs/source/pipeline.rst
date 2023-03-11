.. _pipeline:

Default pipeline
================

Refer to the :py:mod:`.pipeline` module for a general overview of the pipeline concept (involving different stages, inputs, and outputs).

.. _pipeline_theory:

Theory
------

Deformable shape models
^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\omega \subseteq \Omega` be any image region, that is a non-empty subset of the image points :math:`\Omega` in an arbitrary but fixed order :math:`\omega = \left\{ x_1, \dots, x_{\#\omega} \right\}`. Then, a
*deformable shape model* within this image region is defined as the zero-level set of the deformable surface

.. math:: S_\omega(x; \theta, \xi) = F_\omega^\top \theta + G_\omega \xi,

where

.. math:: F_\omega = \begin{bmatrix} f_{x^{(1)}} & \dots & f_{x^{(\#\omega)}} \end{bmatrix},

:math:`f_x` is a second-order polynomial basis function expansion of the image point :math:`x`, and :math:`G_\omega` is a block Toeplitz matrix where each row corresponds to a Gaussian function with standard deviation :math:`\sigma_G` centered at the image points :math:`x_1, \dots, x_{\#\omega}`. The vectors :math:`\theta` and :math:`\xi` are the polynomial parameters and the deformation parameters, respectively. See Section 2.1 of the paper for more details.

Convex energy minimization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Model fitting within any image region :math:`\omega` is performed by minimization of the *convex* energy function,

.. math:: \psi_\omega(\theta, \xi) = \ell(\theta, \xi) + \alpha \cdot \|\xi\|_1

where :math:`\ell(\theta, \xi)` is a *convex* loss function defined by

.. math:: \ell(\theta, \xi) = \mathbb 1^\top_{\#\omega} \ln(1 + \exp(-Y_\omega \cdot S_\omega(\theta, \xi))).

See Section 2.2 of the paper for more details.

The vector :math:`Y_\omega` corresponds to the image intensities, shifted by the intensity offsets :math:`\tau_{x^{(1)}}, \dots, \tau_{x^{(\#\omega)}}`. These offsets are chosen so that they *roughly* separate image foreground and image background, in the sense that image foreground *rather* corresponds to positive components of the vector

.. math:: Y_\omega^\top = \begin{bmatrix} g_{x^{(1)}} - \tau_{x^{(1)}} & \dots & g_{x^{(\#\omega)}} - \tau_{x^{(\#\omega)}} \end{bmatrix},

whereas image background *rather* corresponds to negative components. See Supplemental Material 1 of the paper for more details.

.. _pipeline_stages:

Pipeline stages
---------------

The function :py:meth:`.pipeline.create_default_pipeline` employs the following stages:

#. :py:class:`~.preprocess.Preprocessing` — Implements the computation of the intensity offsets.
#. :py:class:`~.dsmcfg.DSM_ConfigStage` — Provides the hyperparameters from the ``dsm`` namespace as an output.
#. :py:class:`~.c2freganal.C2F_RegionAnalysis` — Implements the coarse-to-fine region analysis scheme.
#. :py:class:`~.globalenergymin.GlobalEnergyMinimization` — Implements the global energy minimization.
#. :py:class:`~.postprocess.Postprocessing` — Discards spurious objects and refines the segmentation masks.

.. _pipeline_inputs_and_outputs:

Inputs and outputs
------------------

Pipeline stages require different inputs and produce different outputs. Below is an overview over all inputs and outputs available within the default pipeline:

``g_raw``
    The raw image intensities. This is the normalized original image, unless histological image data is being processed (i.e. the hyperparameter ``histological`` is set to ``True``). Provided by the pipeline via the :py:meth:`~.pipeline.Pipeline.init` method.

``g_rgb``
    This is the original image, if histological image data is being processed (i.e. the hyperparameter ``histological`` is set to ``True``). Otherwise, ``g_rgb`` is not available as an input. Provided by the pipeline via the :py:meth:`~.pipeline.Pipeline.init` method.

``y``
    The offset image intensities (object of type ``numpy.ndarray`` of the same shape as the ``g_raw`` image). Corresponds to :math:`Y_\Omega` in the paper (see :ref:`Eq. (5) in Section 2.2 <references>`). Provided by the :py:class:`~.preprocess.Preprocessing` stage.

``dsm_cfg``
    A dictionary corresponding to the hyperparameters which reside in the ``dsm`` namespace. Provided by the :py:class:`~.dsmcfg.DSM_ConfigStage` stage.

``y_mask``
    Binary image corresponding to a mask of "empty" image regions (``False``), that are discarded from consideration, and those which possibly contain objects and are considered for segmentation (``True``). This is described in :ref:`Section 3.1 of the paper <references>`. Provided by the :py:class:`~.dsmcfg.C2F_RegionAnalysis` stage.

``g_atoms``
    Integer-valued image representing the universe of atomic image regions (see :ref:`Section 2.3 of the paper <references>`). Provided by the :py:class:`~.dsmcfg.C2F_RegionAnalysis` stage.

``adjacencies``
    The adjacencies of the atomic image regions, represented as an object of the type :py:class:`~.atoms.AtomAdjacencyGraph`. This corresponds to the adjacency graph :math:`\mathcal G` as defined in :ref:`Definition 1 in the paper <references>`. Provided by the :py:class:`~.dsmcfg.C2F_RegionAnalysis` stage.

``seeds``
    The seed points which were used by the Algorithm S1 (described in :ref:`Supplemental Material 5 of the paper <references>`) to determine the atomic image regions, represented by a list of tuples of coordinates. Provided by the :py:class:`~.dsmcfg.C2F_RegionAnalysis` stage.

``clusters``
    Integer-valued image representing the regions of possibly clustered obejcts (see :ref:`Section 2.3 of the paper <references>`). Provided by the :py:class:`~.dsmcfg.C2F_RegionAnalysis` stage.

``y_img``
    An :py:class:`~.image.Image` object corresponding to a joint representation of the offset image intensities ``y`` and mask ``y_mask``. Provided by the :py:class:`~.globalenergymin.GlobalEnergyMinimization` stage.

``cover``
    An :py:class:`~.minsetcover.MinSetCover` object corresponding to :math:`\operatorname{MSC}(\mathscr U_{\# U})` in the paper (see :ref:`Section 2.3.3 <references>`). The solution is accessible via its :py:attr:`~.minsetcover.MinSetCover.solution` property. Provided by the :py:class:`~.globalenergymin.GlobalEnergyMinimization` stage.

``objects``
    List of all computed objects, each represented by the :py:class:`~.objects.Object` class. Corresponds to :math:`\mathscr U_{\# U}` in the paper (see :ref:`Section 2.3.3 <references>`). Provided by the :py:class:`~.globalenergymin.GlobalEnergyMinimization` stage.

``workload``
    The cardinality of the set of all possible objects. Corresponds to the cardinality of :math:`\mathbb P(U)` in the paper (see :ref:`Eq. (9) in Section 2.3.1 <references>`). Provided by the :py:class:`~.globalenergymin.GlobalEnergyMinimization` stage.

``postprocessed_objects``
    List of post-processed objects, each represented by the :py:class:`~.postprocess.PostprocessedObject` class. Provided by the :py:class:`~.postprocess.Postprocessing` stage.
    

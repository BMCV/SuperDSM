.. _pipeline:

Default pipeline
================

Refer to the :py:mod:`.pipeline` module for a general overview of the pipeline concept (involving different stages, inputs, and outputs).

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
    A dictionary corresponding to the hyperparameters which reside in the ``dsm`` namespace. Provided by the :py:class:`~.dsmcfg.PreproDSM_ConfigStagecessing` stage.

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
    
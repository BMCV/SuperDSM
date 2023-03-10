.. _pipeline:

Default pipeline
================

Refer to the :py:mod:`superdsm.pipeline` module for a general overview of the pipeline concept (involving different stages, inputs, and outputs).

.. _pipeline_stages:

Pipeline stages
---------------

The function :py:meth:`superdsm.pipeline.create_default_pipeline` employs the following stages:

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
    The raw image intensities. This is the normalized original image, unless histological image data is being processed (i.e. the hyperparameter ``histological`` is set to ``True``). Provided by the pipeline via the :py:meth:`~superdsm.pipeline.Pipeline.init` method.

``g_rgb``
    This is the original image, if histological image data is being processed (i.e. the hyperparameter ``histological`` is set to ``True``). Otherwise, ``g_rgb`` is not available as an input. Provided by the pipeline via the :py:meth:`~superdsm.pipeline.Pipeline.init` method.

``y``
    The offset image intensities (object of type ``numpy.ndarray`` of the same shape as the ``g_raw`` image). Corresponds to :math:`Y_\Omega` in the paper (:ref:`Kostrykin and Rohr, 2023 <references>`, see Eq. (5) in Section 2.2). Provided by the :py:class:`~.preprocess.Preprocessing` stage.

``dsm_cfg``
    An object of type :py:class:`~superdsm.config.Config` corresponding to the hyperparameters which reside in the ``dsm`` namespace. Provided by the :py:class:`~.dsmcfg.PreproDSM_ConfigStagecessing` stage.

``y_mask``
    tbd.

``g_atoms``
    tbd.

``adjacencies``
    tbd.

``seeds``
    tbd.

``clusters``
    tbd.

``y_img``
    tbd.

``cover``
    tbd.

``objects``
    tbd.

``workload``
    tbd.

``postprocessed_objects``
    tbd.
    
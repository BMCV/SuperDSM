.. _pipeline:

Default pipeline
================

Refer to the :py:mod:`superdsm.pipeline` module for a general overview of the pipeline concept (involving different stages, inputs, and outputs).

.. _pipeline_stages:

Pipeline stages
---------------

The function :py:meth:`superdsm.pipeline.create_default_pipeline` employs the following stages:

#. :py:class:`~.preprocess.Preprocessing` — Implements the computation of the intensity offsets as described in Supplemental Material 1 of the paper (:ref:`Kostrykin and Rohr, 2023 <references>`).
#. :py:class:`~.dsmcfg.DSM_ConfigStage` — Fetches the hyperparameters from the ``dsm`` namespace and provides them as an output.
#. :py:class:`~.c2freganal.C2F_RegionAnalysis` — Implements the coarse-to-fine region analysis scheme (see Section 3.2 and Supplemental Material 5 of the paper).
#. :py:class:`~.globalenergymin.GlobalEnergyMinimization` — Implements the global energy minimization (see Sections 2.3 and 3.3 of the paper).
#. :py:class:`~.postprocess.Postprocessing` — Discards spurious objects and refines the segmentation masks (see Section 3.4 and Supplemental Material 7 of the paper).

.. _pipeline_inputs_and_outputs:

Inputs and outputs
------------------

tbd.

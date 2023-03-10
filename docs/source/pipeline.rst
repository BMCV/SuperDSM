.. _pipeline:

Default pipeline
================

Refer to the :py:class:`superdsm.pipeline.Pipeline` class for a general overview of the pipeline concept.

.. _pipeline_stages:

Pipeline stages
---------------

The pipeline created by :py:meth:`superdsm.pipeline.create_default_pipeline` consists of the following stages:

#. :py:class:`~.preprocess.Preprocessing`
#. :py:class:`~.dsmcfg.DSM_ConfigStage`
#. :py:class:`~.c2freganal.C2F_RegionAnalysis`
#. :py:class:`~.globalenergymin.GlobalEnergyMinimization`
#. :py:class:`~.postprocess.Postprocessing`

.. _pipeline_inputs_and_outputs:

Inputs and outputs
------------------

tbd.

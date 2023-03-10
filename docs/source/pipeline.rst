.. _pipeline:

Default pipeline
================

Refer to the :py:mod:`superdsm.pipeline` module for a general overview of the pipeline concept.

.. _pipeline_stages:

Pipeline stages
---------------

The function :py:meth:`superdsm.pipeline.create_default_pipeline` employs the following stages:

#. :py:class:`~.preprocess.Preprocessing`
#. :py:class:`~.dsmcfg.DSM_ConfigStage`
#. :py:class:`~.c2freganal.C2F_RegionAnalysis`
#. :py:class:`~.globalenergymin.GlobalEnergyMinimization`
#. :py:class:`~.postprocess.Postprocessing`

.. _pipeline_inputs_and_outputs:

Inputs and outputs
------------------

tbd.

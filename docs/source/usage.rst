Quick start
===========

.. _installation:

Installation
------------

To use SuperDSM, first install it using conda:

.. code-block:: console

   conda install -c bioconda superdsm

Usage
-----

.. _usage_example_batch:

Batch processing
****************

To run SuperDSM from command line, use:

.. code-block:: console

   python -m 'superdsm.batch' --help

To export (intermediate) results, use:

.. code-block:: console

   python -m 'superdsm.export' --help

For details, see :ref:`batch_system`.

.. _usage_example_interactive:

Interactive
***********

To use SuperDSM interactively, i.e. programatically as opposed to batch processing, the first step is to `initialize Ray <https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html>`_. This is the multiprocessing framework used by SuperDSM. Initialization is simple, just remember to adapt the number of CPUs to be used:

.. code-block:: python

   import ray
   ray.init(num_cpus=16, log_to_driver=False, logging_level='error')

After this initialization routine, SuperDSM is ready to use:

.. code-block:: python

    import superdsm.automation
    pipeline = superdsm.pipeline.create_default_pipeline()
    cfg = superdsm.config.Config()
    data, _, _ = superdsm.automation.process_image(pipeline, cfg, img)

In this example, the default set of hyperparameters will be used. The parameters can be changed using the ``cfg`` object (see the :py:class:`~superdsm.config.Config` class API). The available hyperparameters are described in the documentation of the respective stages employed in the pipeline created by the :py:meth:`~superdsm.pipeline.create_default_pipeline` function.

The variable ``img`` must be a two-dimensional ``numpy.ndarray`` object which represents the raw image intensities. Images can be loaded from file using :py:meth:`~superdsm.io.imread`.

The pipeline data object ``data`` is a dictionary containing all the intermediate results which might be necessary for further computations. This can also be used to obtain a graphical representation of the segmentation results:

.. code-block:: python

    import superdsm.render
    seg = superdsm.render.render_result_over_image(data)

The ``seg`` object returned by the :py:meth:`~superdsm.render.render_result_over_image` function is an RGB image (represented by a ``numpy.ndarray`` object) which can be visualized directly (e.g., using matplotlib) or saved for later use (e.g., using :py:meth:`~superdsm.io.imwrite`). Use :py:meth:`~superdsm.render.rasterize_labels` to obtain segmentation masks from the pipeline data object.

.. _env_variables:

Environment variables
*********************

MKL_DEBUG_CPU_TYPE

    To take advantage of the acceleration provided by MKL on AMD CPUs, the environment variable ``MKL_DEBUG_CPU_TYPE=5`` should be set when using an AMD CPU. This usually happens automatically, unless automatic recognition of the CPU vendor fails (and a warning is shown).

SUPERDSM_INTERMEDIATE_OUTPUT

   Set ``SUPERDSM_INTERMEDIATE_OUTPUT=0`` to mute the intermediate console output.

SUPERDSM_NUM_CPUS

   Set ``SUPERDSM_NUM_CPUS=8`` to use 8 CPU cores in batch processing. Defaults to 2. Ignored when used interactively.

.. _references:

References
----------

If you use SuperDSM, please cite:

* L\. Kostrykin and K\. Rohr, *"Robust Graph Pruning for Efficient Segmentation and Cluster Splitting of Cell Nuclei using Deformable Shape Models,"* accepted for presentation at *IEEE International Symposium on Biomedical Imaging (ISBI)*, Athens, Greece, May 27–30, 2024.

* L\. Kostrykin and K\. Rohr, *"Superadditivity and Convex Optimization for Globally Optimal Cell Segmentation Using Deformable Shape Models,"* in *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 45(3), pp. 3831–3847, 2023.
  `[doi] <https://doi.org/10.1109/TPAMI.2022.3185583>`_

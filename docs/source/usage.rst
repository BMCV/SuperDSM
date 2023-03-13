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

For details, see :ref:`batch_processing`.

.. _usage_example_interactive:

Interactive
***********

To use SuperDSM interactively, i.e. programatically, as opposed to batch processing, the first step is to `initialize Ray <https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html>`_. This is the multiprocessing framework used by SuperDSM. Initialization is simple, just remember to adapt the number of CPUs to be used:

.. code-block:: python

   import ray
   ray.init(num_cpus=16, log_to_driver=False, logging_level=ray.logging.ERROR)

After this initialization routine, SuperDSM is ready to use:

.. code-block:: python

    import superdsm.automation
    pipeline = superdsm.pipeline.create_default_pipeline()
    cfg = superdsm.config.Config()
    results, _, _ = superdsm.automation.process_image(pipeline, cfg, img)

In this example, the default set of hyperparameters will be used. The parameters can be changed using the ``cfg`` object (see the :py:class:`~superdsm.config.Config` class API). The available hyperparameters are described in the documentation of the respective stages employed in the pipeline created by the :py:meth:`~superdsm.pipeline.create_default_pipeline` function.

The variable ``img`` must be a two-dimensional ``numpy.ndarray`` object which represents the raw image intensities. Images can be loaded from file using :py:meth:`~superdsm.io.imread`.

The dictionary ``results`` contains all the intermediate results which might be necessary for further computations. This can also be used to obtain a graphical representation of the segmentation results:

.. code-block:: python

    import superdsm.render
    result_img = superdsm.render.render_result_over_image(results)

The obtained ``result_img`` object is an RGB image (represented by a ``numpy.ndarray`` object) which can be visualized directly (e.g., using matplotlib) or saved for later use (e.g., using :py:meth:`~superdsm.io.imwrite`).

Use :py:meth:`~superdsm.render.rasterize_labels` to obtain segmentation masks from the ``results`` dictionary.

.. _references:

References
----------

If you use SuperDSM, please cite:

`L. Kostrykin and K. Rohr, "Superadditivity and Convex Optimization for Globally Optimal Cell Segmentation Using Deformable Shape Models," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45(3), pp. 3831–3847, 2023.
<https://doi.org/10.1109/TPAMI.2022.3185583>`_

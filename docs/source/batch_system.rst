.. _batch_system:

Batch system
============

Task specification
------------------

To perform batch processing of a dataset, you first need to create a *task*. To do that, create an empty directory, and put a ``task.json`` file in it. This file will contain the specification of the segmentation task. Below is an example specification:

.. code-block:: json

   {
       "runnable": true,
       "num_cpus": 16,
       "environ": {
           "MKL_NUM_THREADS": 2,
           "OPENBLAS_NUM_THREADS": 2
       },

       "img_pathpattern": "/data/dataset/img-%d.tiff",
       "seg_pathpattern": "seg/dna-%d.png",
       "adj_pathpattern": "adj/dna-%d.png",
       "log_pathpattern": "log/dna-%d",
       "cfg_pathpattern": "cfg/dna-%d.json",
       "overlay_pathpattern": "overlays/dna-%d.png",
       "file_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

       "config": {
       }
   }

The meaning of the different fields is the follows:

``runnable``
    Marks this task as runnable (or not runnable). If set to ``false``, the specification will be treated as a template for derived tasks. Derived tasks are placed in sub-folders and inherit the specification of the parent task. This is useful, for example, if you want to try out different hyperparameters. The batch system automatically picks up intermediate results of parent tasks to speed up the completion of derived tasks.

``num_cpus``
    The number of processes which is to be used simultaneously (in parallel).

``environ``
    Defines environment variables which are to be set. In the example above, MKL and OpenBLAS numpy backends are both instructed to use two threads for parallel computations.

``img_pathpattern``
    Defines the path to the input images of the dataset, using placeholders like ``%d`` for decimals and ``%s`` for strings (decimals can also be padded with zeros to a fixed length using, e.g., use ``%02d`` for a length of 2).

``seg_pathpattern``
    Relative path of files, where the segmentation masks are to be written to, using placeholders as described above.

``adj_pathpattern``
    Relative path of files, where the images of the atomic image regions and adjacency graphs are to be written to, using placeholders as described above (see :ref:`pipeline_theory_c2freganal`).

``log_pathpattern``
    Relative path of files, where the logs are to be written to, using placeholders as described above (mainly for debugging purposes).

``cfg_pathpattern``
    Relative path of files, where the hyperparameters are to be written to, using placeholders as described above (mainly for reviewing the automatically generated hyperparameters).

``file_ids``
    List of file IDs, which are used to resolve the pattern-based fields described above. In the considered example, the list of input images will resolve to ``/data/dataset/img-1.tiff``, …, ``/data/dataset/img-10.tiff``. File IDs are allowed to be strings, and they are also allowed to contain ``/`` to encode paths which involve sub-directories.

``config``
    Defines the hyperparameters to be used. The available hyperparameters are described in the documentation of the respective stages of the default pipeline (see :ref:`pipeline_stages`). Note that namespaces must be specified as nested JSON objects.

Instead of specifying the hyperparameters in the task specification directly, it is also possible to include them from a separate JSON file using the ``base_config_path`` field. The path must be either absolute or relative to the ``task.json`` file. It is also possible to use ``{DIRNAME}`` as a substitute for the name of the directory, which the ``task.json`` file resides in. The placeholder ``{ROOTDIR}`` in the path specification resolves to the *root directory* passed to the batch system (see below).

Examples can be found in the ``examples`` sub-directory of the `SuperDSM repository <https://github.com/BMCV/SuperDSM>`_.

Batch processing
----------------

To perform batch processing of all tasks specified in the current working directory, including all sub-directories and so on:

.. code-block:: console

   python -m 'superdsm.batch' .

This will run the batch system in *dry mode*, so nothing will actually be processed. Instead, each task which is going to be processed will be printed, along with some additional information. To actually start the processing, re-run the command and include the ``--run`` argument.

In this example, the current working directory will correspond to the *root directory* when it comes to resolving the ``{ROOTDIR}`` placeholder in the path specification.

Note that the batch system will automatically skip tasks which already have been completed in a previous run, unless the ``--force`` argument is used. On the other hand, tasks will not be marked as completed if the ``--oneshot`` argument is used. To run only a single task from the root directory, use the ``--task`` argument, or ``--task-dir`` if you want to automatically include the dervied tasks.

Refer to ``python -m 'superdsm.batch' --help`` for further information.
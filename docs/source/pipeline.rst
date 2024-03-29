.. _pipeline:

Default pipeline
================

Refer to the :py:mod:`.pipeline` module for a general overview of the pipeline concept (involving different stages, inputs, and outputs).

.. _pipeline_theory:

Theory
------

This is an overview of the fundamental concepts described in :ref:`Kostrykin and Rohr (TPAMI 2023) <references>`.

.. _pipeline_theory_dsm:

Deformable shape models
^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\omega \subseteq \Omega` be any image region, that is a non-empty subset of the image points :math:`\Omega` in an arbitrary but fixed order :math:`\omega = \left\{ x_1, \dots, x_{\#\omega} \right\}`. Then, a
*deformable shape model* within this image region is defined as the zero-level set of the deformable surface

.. math:: S_\omega(\theta, \xi) = F_\omega^\top \theta + G_\omega \xi,

where

.. math:: F_\omega = \begin{bmatrix} f_{x^{(1)}} & \dots & f_{x^{(\#\omega)}} \end{bmatrix},

:math:`f_x` is a second-order polynomial basis function expansion of the image point :math:`x`, and :math:`G_\omega` is a block Toeplitz matrix where each row corresponds to a Gaussian function with standard deviation :math:`\sigma_G` centered at the image points :math:`x_1, \dots, x_{\#\omega}`. The vectors :math:`\theta` and :math:`\xi` are the polynomial parameters and the deformation parameters, respectively. See Section 2.1 of the paper for more details.

.. _pipeline_theory_cvxprog:

Convex energy minimization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Model fitting within any image region :math:`\omega` is performed by minimization of the *convex* energy function

.. math:: \psi_\omega(\theta, \xi) = \ell(\theta, \xi) + \alpha \cdot \|\xi\|_1,

where :math:`\ell(\theta, \xi)` is a *convex* loss function defined by

.. math:: \ell(\theta, \xi) = \mathbb 1^\top_{\#\omega} \ln(1 + \exp(-Y_\omega \cdot S_\omega(\theta, \xi)))

and :math:`\alpha` is a regularization parameter which governs the regularization of the deformations. This is implemented in the :py:mod:`superdsm.dsm` module. See Section 2.2 of the paper for more details.

The vector :math:`Y_\omega` corresponds to the image intensities, shifted by the intensity offsets :math:`\tau_{x^{(1)}}, \dots, \tau_{x^{(\#\omega)}}`. These offsets are chosen so that they *roughly* separate image foreground and image background, in the sense that image foreground *rather* corresponds to positive components of the vector

.. math:: Y_\omega^\top = \begin{bmatrix} g_{x^{(1)}} - \tau_{x^{(1)}} & \dots & g_{x^{(\#\omega)}} - \tau_{x^{(\#\omega)}} \end{bmatrix},

whereas image background *rather* corresponds to negative components. The computation of the intensity offsets is based on the Gaussian filter :math:`\mathcal G_\sigma` and described in Supplemental Material 1 of the paper.

.. _pipeline_theory_c2freganal:

Coarse-to-fine region analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`U` be a universe of atomic image regions, so that no atomic image region contains more than a single object (but any object can be split into multiple atomic regions). The atomic image regions are generated by recursively splitting image regions until certain criteria are met (the procedure is hence refered to as *coarse-to-fine region analysis*). Image regions are split by choosing two *seed points*, which correspond to local intensity peaks, and performing a seeded watershed transform of the image intensities. Details are given in Supplemental Material 5.

Splitting of image regions is performed according to the *normalized energy*

.. math:: r(\omega) = \inf_\theta \psi_\omega(\theta, \mathbb 0) / \#\omega,

see the :py:class:`~.c2freganal.C2F_RegionAnalysis` stage for details.

Two atomic image regions :math:`u,v \in U` are called *adjacent* if and only if there exists a path :math:`\pi \subset \Omega` between :math:`u` and :math:`v` so that :math:`Y_\omega|_{\omega=\pi} > 0`. Let :math:`\Pi \subseteq U \times U` be the set of all *connected* atomic image regions, i.e. :math:`(u,v) \in \Pi` if and only if the adjacency graph :math:`\mathcal G = (U, \mathcal E)` contains a path between :math:`u` and :math:`v`. Details are given in Section 2.3.1 of the paper.

.. _pipeline_theory_jointsegandclustersplit:

Joint segmentation and cluster splitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Global energy minimization is performed by solving :math:`\operatorname{MSC}(\mathbb P(U))`, where

.. math:: \mathbb P(U) = \{ X \subseteq U | X \neq \emptyset, X \times X \subseteq \Pi \}

and

.. math:: \operatorname{MSC}(\mathscr S) = \min_{\mathscr X \subseteq \mathscr S} \sum_{X \in \mathscr X} \beta + \nu(X) \enspace\text{s.t. } \bigcup \mathscr S = \bigcup \mathscr X

is an instance of the *min-weight set-cover* problem, and

.. math:: \nu(X) = \inf_{\theta,\xi} \psi_\omega(\theta,\xi)|_{\omega = \bigcup X}

is the *set energy function*. The constant term :math:`\beta` governs the sparsity of the solution. It is also the maximum allowed energy difference of merging two deformable shape models (two image regions). See Section 2.3.2 of the paper for details.

Instead of solving :math:`\operatorname{MSC}(\mathbb P(U))` directly, a sequence :math:`\mathscr U_1, \dots, \mathscr U_{\# U} \subseteq \mathbb P(U)` is computed so that

.. math:: \operatorname{MSC}(\mathbb P(U)) = \operatorname{MSC}(\mathscr U_{\# U}).

If, however, :math:`c(U) \leq \beta + \sum_{u \in U} c(\{u\})`, then the closed-form solution

.. math:: \operatorname{MSC}(\mathbb P(U)) = c(U)

holds and the sequential computation is not required. Regions of possibly clustered objects are processed separately of each other, so, in fact, there are multiple disjoint universes of atomic image regions per image. Thus, the closed-form solution corresponds to cases of non-clustered objects. See Sections 2.3.3, 3.1, and 3.3 of the paper for details.

.. _pipeline_stages:

Pipeline stages
---------------

The function :py:meth:`pipeline.create_default_pipeline() <superdsm.pipeline.create_default_pipeline>` employs the following stages:

#. :py:class:`~.preprocess.Preprocessing` — Implements the computation of the intensity offsets.
#. :py:class:`~.dsmcfg.DSM_Config` — Provides the hyperparameters from the ``dsm`` namespace as an output.
#. :py:class:`~.c2freganal.C2F_RegionAnalysis` — Implements the coarse-to-fine region analysis scheme.
#. :py:class:`~.globalenergymin.GlobalEnergyMinimization` — Implements the global energy minimization.
#. :py:class:`~.postprocess.Postprocessing` — Discards spurious objects and refines the segmentation masks.

.. _pipeline_inputs_and_outputs:

Inputs and outputs
------------------

Pipeline stages require different inputs and produce different outputs. These are like intermediate results, which are shared or passed between the stages. The pipeline maintains their state, which is kept inside the *pipeline data object*. Below is an overview over all inputs and outputs available within the default pipeline:

``g_raw``
    The raw image intensities :math:`g_{x^{1}}, \dots, g_{x^{\#\Omega}}`, normalized so that the intensities range from 0 to 1. Up to the normalization, this corresponds to the original input image, unless histological image data is being processed (i.e. the hyperparameter ``histological`` is set to ``True``). Provided by the pipeline via the :py:meth:`~.pipeline.Pipeline.init` method, refer to its documentation for details.

``g_rgb``
    This is the original image, if histological image data is being processed (i.e. the hyperparameter ``histological`` is set to ``True``). Otherwise, ``g_rgb`` is not available as an input. Provided by the pipeline via the :py:meth:`~.pipeline.Pipeline.init` method, refer to its documentation for details.

``y``
    The offset image intensities :math:`Y_\omega|_{\omega = \Omega}`, represented as an object of type ``numpy.ndarray`` of the same shape as the ``g_raw`` image. Provided by the :py:class:`~.preprocess.Preprocessing` stage.

``dsm_cfg``
    A dictionary corresponding to the hyperparameters which reside in the ``dsm`` namespace. Provided by the :py:class:`~.dsmcfg.DSM_Config` stage.

``y_mask``
    Binary image corresponding to a mask of "empty" image regions (``False``), that are discarded from consideration, and those which possibly contain objects and are considered for segmentation (``True``). This is described in Section 3.1 of the paper. Provided by the :py:class:`~.c2freganal.C2F_RegionAnalysis` stage.

``atoms``
    Integer-valued image representing the universe of atomic image regions. Each atomic image region has a unique label, which is the integer value. Provided by the :py:class:`~.c2freganal.C2F_RegionAnalysis` stage.

``adjacencies``
    The adjacency graph :math:`\mathcal G`, represented as an object of the type :py:class:`~.atoms.AtomAdjacencyGraph`. Provided by the :py:class:`~.c2freganal.C2F_RegionAnalysis` stage.

``seeds``
    The seed points which were used to determine the atomic image regions, represented by a list of tuples of coordinates. Provided by the :py:class:`~.c2freganal.C2F_RegionAnalysis` stage.

``clusters``
    Integer-valued image representing the regions of possibly clustered obejcts. Each region has a unique label, which is the integer value. Provided by the :py:class:`~.c2freganal.C2F_RegionAnalysis` stage.

``y_img``
    An :py:class:`~.image.Image` object corresponding to a joint representation of the offset image intensities ``y`` and mask ``y_mask``. Provided by the :py:class:`~.globalenergymin.GlobalEnergyMinimization` stage.

``cover``
    An :py:class:`~.minsetcover.MinSetCover` object corresponding to :math:`\operatorname{MSC}(\mathscr U_{\# U})`. The optimal family :math:`\mathscr X \subseteq \mathbb P(U)` is accessible via its :py:attr:`~.minsetcover.MinSetCover.solution` property. Provided by the :py:class:`~.globalenergymin.GlobalEnergyMinimization` stage.

``objects``
    List of all computed objects :math:`\mathscr U_{\# U}`, each represented by the :py:class:`~.objects.Object` class. Provided by the :py:class:`~.globalenergymin.GlobalEnergyMinimization` stage.

``performance``
    An object of the :py:class:`~.globalenergymin.PerformanceReport` class which carries values indicating the performance of the algorithms used by the :py:class:`~.globalenergymin.GlobalEnergyMinimization` stage. Provided by the :py:class:`~.globalenergymin.GlobalEnergyMinimization` stage.

``postprocessed_objects``
    List of post-processed objects, each represented by the :py:class:`~.postprocess.PostprocessedObject` class. Provided by the :py:class:`~.postprocess.Postprocessing` stage.
    
.. _batch_system:

Batch system
------------

.. _batch_task_spec:

Task specification
^^^^^^^^^^^^^^^^^^

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

``last_stage``
    If specified, then the pipeline processing will end at the specified stage.

``dilate``
    Performs morphological dilation for all final segmentation masks, using the given amount of pixels. For negative values, morphological erosion is performed.

``merge_overlap_threshold``
    If specified, then any pair of two objects (final segmentation masks) with an overlap larger than this threshold will be merged into a single object.

``config``
    Defines the hyperparameters to be used. The available hyperparameters are described in the documentation of the respective stages of the default pipeline (see :ref:`pipeline_stages`). Note that namespaces must be specified as nested JSON objects.

Instead of specifying the hyperparameters in the task specification directly, it is also possible to include them from a separate JSON file using the ``base_config_path`` field. The path must be either absolute or relative to the ``task.json`` file. It is also possible to use ``{DIRNAME}`` as a substitute for the name of the directory, which the ``task.json`` file resides in. The placeholder ``{ROOTDIR}`` in the path specification resolves to the *root directory* passed to the batch system (see below).

Examples can be found in the ``examples`` sub-directory of the `SuperDSM repository <https://github.com/BMCV/SuperDSM>`_.

.. _batch_prcessing:

Batch processing
^^^^^^^^^^^^^^^^

To perform batch processing of all tasks specified in the current working directory, including all sub-directories and so on:

.. code-block:: console

   python -m 'superdsm.batch' .

This will run the batch system in *dry mode*, so nothing will actually be processed. Instead, each task which is going to be processed will be printed, along with some additional information. To actually start the processing, re-run the command and include the ``--run`` argument.

In this example, the current working directory will correspond to the *root directory* when it comes to resolving the ``{ROOTDIR}`` placeholder in the path specification.

Note that the batch system will automatically skip tasks which already have been completed in a previous run, unless the ``--force`` argument is used. On the other hand, tasks will not be marked as completed if the ``--oneshot`` argument is used. To run only a single task from the root directory, use the ``--task`` argument, or ``--task-dir`` if you want to automatically include the dervied tasks. Note that, in both cases, the tasks must be specified relatively to the root directory.

Refer to ``python -m 'superdsm.batch' --help`` for further information.

SuperDSM
==========

SuperDSM is a globally optimal segmentation method based on superadditivity and deformable shape models for cell nuclei in fluorescence microscopy images and beyond.

The documentation is available here: https://superdsm.readthedocs.io

Use ``python -m unittest`` in the root directory to run the test suite.

For full performance on both Intel and AMD CPUs, NumPy with MKL support must be used (instead of OpenBLAS which is often the default). When using the source tree instead of the Conda package from Bioconda, this can be ensured by adding the dependency ``blas =*=mkl`` to your Conda environment.

**Publications:**

* L\. Kostrykin and K\. Rohr, *"Robust Graph Pruning for Efficient Segmentation and Cluster Splitting of Cell Nuclei using Deformable Shape Models,"* accepted for presentation at *IEEE International Symposium on Biomedical Imaging (ISBI)*, Athens, Greece, May 27–30, 2024.

* L\. Kostrykin and K\. Rohr, *"Superadditivity and Convex Optimization for Globally Optimal Cell Segmentation Using Deformable Shape Models,"* in *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 45(3), pp. 3831–3847, 2023.
  `[doi] <https://doi.org/10.1109/TPAMI.2022.3185583>`_

----

Copyright (c) 2017-2024 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University

This work is licensed under the terms of the MIT license.
For a copy, see `LICENSE </LICENSE>`_.

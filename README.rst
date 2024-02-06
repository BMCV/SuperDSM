`SuperDSM <https://github.com/BMCV/SuperDSM>`_
==============================================

.. image:: https://img.shields.io/badge/Install%20with-conda-%2387c305
   :target: https://anaconda.org/bioconda/superdsm

.. image:: https://img.shields.io/conda/v/bioconda/superdsm.svg?label=Version
   :target: https://anaconda.org/bioconda/superdsm

.. image:: https://img.shields.io/conda/dn/bioconda/superdsm.svg?label=Downloads
   :target: https://anaconda.org/bioconda/superdsm
    
.. image:: https://readthedocs.org/projects/superdsm/badge/?version=latest
   :target: https://superdsm.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/usegalaxy-.eu-brightgreen?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAASCAYAAABB7B6eAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAACXBIWXMAAAsTAAALEwEAmpwYAAACC2lUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iWE1QIENvcmUgNS40LjAiPgogICA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgogICAgICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyI+CiAgICAgICAgIDx0aWZmOlJlc29sdXRpb25Vbml0PjI8L3RpZmY6UmVzb2x1dGlvblVuaXQ+CiAgICAgICAgIDx0aWZmOkNvbXByZXNzaW9uPjE8L3RpZmY6Q29tcHJlc3Npb24+CiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24+CiAgICAgICAgIDx0aWZmOlBob3RvbWV0cmljSW50ZXJwcmV0YXRpb24+MjwvdGlmZjpQaG90b21ldHJpY0ludGVycHJldGF0aW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KD0UqkwAAAn9JREFUOBGlVEuLE0EQruqZiftwDz4QYT1IYM8eFkHFw/4HYX+GB3/B4l/YP+CP8OBNTwpCwFMQXAQPKtnsg5nJZpKdni6/6kzHvAYDFtRUT71f3UwAEbkLch9ogQxcBwRKMfAnM1/CBwgrbxkgPAYqlBOy1jfovlaPsEiWPROZmqmZKKzOYCJb/AbdYLso9/9B6GppBRqCrjSYYaquZq20EUKAzVpjo1FzWRDVrNay6C/HDxT92wXrAVCH3ASqq5VqEtv1WZ13Mdwf8LFyyKECNbgHHAObWhScf4Wnj9CbQpPzWYU3UFoX3qkhlG8AY2BTQt5/EA7qaEPQsgGLWied0A8VKrHAsCC1eJ6EFoUd1v6GoPOaRAtDPViUr/wPzkIFV9AaAZGtYB568VyJfijV+ZBzlVZJ3W7XHB2RESGe4opXIGzRTdjcAupOK09RA6kzr1NTrTj7V1ugM4VgPGWEw+e39CxO6JUw5XhhKihmaDacU2GiR0Ohcc4cZ+Kq3AjlEnEeRSazLs6/9b/kh4eTC+hngE3QQD7Yyclxsrf3cpxsPXn+cFdenF9aqlBXMXaDiEyfyfawBz2RqC/O9WF1ysacOpytlUSoqNrtfbS642+4D4CS9V3xb4u8P/ACI4O810efRu6KsC0QnjHJGaq4IOGUjWTo/YDZDB3xSIxcGyNlWcTucb4T3in/3IaueNrZyX0lGOrWndstOr+w21UlVFokILjJLFhPukbVY8OmwNQ3nZgNJNmKDccusSb4UIe+gtkI+9/bSLJDjqn763f5CQ5TLApmICkqwR0QnUPKZFIUnoozWcQuRbC0Km02knj0tPYx63furGs3x/iPnz83zJDVNtdP3QAAAABJRU5ErkJggg==
   :target: https://usegalaxy.eu/root?tool_id=toolshed.g2.bx.psu.edu/repos/imgteam/superdsm/ip_superdsm

SuperDSM is a globally optimal segmentation method based on superadditivity and deformable shape models for cell nuclei in fluorescence microscopy images and beyond.

The documentation is available here: https://superdsm.readthedocs.io

Use ``python -m unittest`` in the root directory of the repository to run the test suite.

Dependency Version Considerations:
""""""""""""""""""""""""""""""""""

The file *superdsm.yml* specifies the Conda environment required to accurately reproduce the results from our publications. For most of the dependencies (maybe even all), newer versions are also known to work, however, it has been observed that using different versions might yield slightly different results. To enhance consistency, reproducibility, and `FAIRness <https://www.nature.com/articles/s41597-022-01710-x>`_, most dependency versions are thus pinned.

This also concerns BLAS, which is pinned to ``blas==1.0``. As an alternative to using Conda, *requirements.txt* specifies the required *pip* dependencies with pinned versions. However, to the best of our knowledge, it is not possible to request a specific BLAS version in *pip*, meaning that using *pip* instead of Conda is discouraged.

Note that our Conda package from Bioconda allows different dependency versions, because otherwise it would not be possible to use the package with newer versions of Python. Thus, when using our Conda package, keep in mind that sticking to the versions of the dependencies specified in *superdsm.yml* is recommended.

Performance Considerations:
"""""""""""""""""""""""""""

For full performance on both Intel and AMD CPUs, NumPy with MKL support must be used (instead of OpenBLAS which is often the default, see `details <https://stackoverflow.com/questions/62783262/why-is-numpy-with-ryzen-threadripper-so-much-slower-than-xeon>`_). When using Conda, this can be ensured by adding the dependency ``blas=1.0=mkl`` to the Conda environment (or ``blas=*=mkl`` to allow different versions).

To take advantage of the acceleration provided by MKL on AMD CPUs, the MKL version must be pinned to ``2020.0``. Both specifications are included in the Conda environment specified in *superdsm.yml*. In addition, the environment variable ``MKL_DEBUG_CPU_TYPE=5`` must be set.

Later versions of MKL do not support ``MKL_DEBUG_CPU_TYPE=5``, and previous versions do not offer the required APIs. Unfortunately, it looks like this particular version of MKL has been removed from PyPI (see `available versions <https://pypi.org/project/mkl/#history>`_), so it is not possible to gain the full performance on AMD CPUs using *pip* instead of Conda, and thus the version of MKL is not pinned in *requirements.txt* by default.

Publications:
"""""""""""""

* L\. Kostrykin and K\. Rohr, *"Robust Graph Pruning for Efficient Segmentation and Cluster Splitting of Cell Nuclei using Deformable Shape Models,"* accepted for presentation at *IEEE International Symposium on Biomedical Imaging (ISBI)*, Athens, Greece, May 27–30, 2024.

* L\. Kostrykin and K\. Rohr, *"Superadditivity and Convex Optimization for Globally Optimal Cell Segmentation Using Deformable Shape Models,"* in *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 45(3), pp. 3831–3847, 2023.
  `[doi] <https://doi.org/10.1109/TPAMI.2022.3185583>`_

----

Copyright (c) 2017-2024 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University

This work is licensed under the terms of the MIT license.
For a copy, see `LICENSE </LICENSE>`_.

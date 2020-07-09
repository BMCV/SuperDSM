import sys
import os.path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sparse_dot'))

import sparse_dot_mkl as mkl

dot  = mkl.dot_product_mkl
gram = mkl.dot_product_transpose_mkl


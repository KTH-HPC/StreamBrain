from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys


# Set backend based on BCPNN_BACKEND flag, if applicable.
if 'BCPNN_BACKEND' in os.environ:
    _BACKEND = os.environ['BCPNN_BACKEND']
else:
    _BACKEND = 'numpy'


# Import backend functions.
if _BACKEND == 'numpy':
    sys.stderr.write('Using Numpy backend\n')
    from .backend.numpy_backend import *
elif _BACKEND == 'cpu':
    sys.stderr.write('Using CPU backend.\n')
    from .backend.cpu_backend import *
elif _BACKEND == 'gpu':
    sys.stderr.write('Using GPU backend.\n')
    from .backend.cuda_backend import *
elif _BACKEND == 'full_cuda':
    sys.stderr.write('Using full cuda backend.\n')
    from .backend.full_cuda_backend import *
elif _BACKEND == 'fpga':
    sys.stderr.write('Using GPU backend.\n')
    from .backend.fpga_backend import *
else:
    raise ValueError('Unable to import backend : ' + str(_BACKEND))


def backend():
    return _BACKEND

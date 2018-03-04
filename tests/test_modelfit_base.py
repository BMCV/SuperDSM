#!/usr/bin/env python2

import sys
import numpy as np
import scipy.ndimage as ndi
import testdata

from math import sqrt

sys.path.append('..')

import gocell.modelfit_base
import gocell.labels


g, groundtruth = testdata.create_surface()
y = gocell.labels.ThresholdedLabels(g, 0.5).get_map()
w = np.ones_like(g.model) / np.prod(g.model.shape)
r = ndi.filters.gaussian_gradient_magnitude(g.model, 5.)


def compute_finite_diff_grad(f, p0, s):
    if not hasattr(s, '__len__'): s = [s] * len(p0)
    pd   = np.zeros_like(p0)
    grad = np.zeros_like(pd)
    for i, si in enumerate(s):
        pd  [i] = si
        grad[i] = (f(p0 + pd) - f(p0 - pd)) / (2 * si)
        pd  [i] = 0
    return grad


def test_finite_diff_grad(f, p0, s):
    grad_expected = compute_finite_diff_grad(f, p0, s)
    grad_actual   = f.grad(p0)
    return sqrt(np.square(grad_expected - grad_actual).mean())


def compute_finite_diff_hessian(f, p0, s):
    if not hasattr(s, '__len__'): s = [s] * len(p0)
    pd = np.zeros_like(p0)
    H  = np.zeros((len(p0), len(p0)))
    for i, si in enumerate(s):
        pd  [i] = si
        H[:, i] = (f.grad(p0 + pd) - f.grad(p0 - pd)) / (2 * si)
        pd  [i] = 0
    return H


def test_finite_diff_hessian(f, p0, s):
    H_expected = compute_finite_diff_hessian(f, p0, s)
    H_actual   = f.hessian(p0)
    return sqrt(np.square(H_expected - H_actual).mean())


sw = 1e-5
for epsilon in [1e-6, 1e-3, 1]:

    print('')
    print('*** Running with epsilon = %f' % epsilon)
    print('')

    J = gocell.modelfit_base.Energy(y, g, w, r, epsilon=epsilon)

    errors = [test_finite_diff_grad(J, groundtruth.array, sw)]
    print('Finite difference testing of gradient computation:')
    print('  at ground truth: %e' % errors[-1])
    
    np.random.seed(0)
    for i in xrange(50):
        errors.append(test_finite_diff_grad(J, np.random.randn(len(groundtruth.array)), sw))
        print('  at random %5s: %e' % ('(%d)' % (i + 1), errors[-1]))
    
    print('\nMaximum error: %e' % max(errors))
    assert max(errors) < 1e-8, 'Gradient validation failed'
    
    
    errors = [test_finite_diff_hessian(J, groundtruth.array, sw)]
    print('Finite difference testing of Hessian computation:')
    print('  at ground truth: %e' % errors[-1])
    
    np.random.seed(0)
    for i in xrange(50):
        errors.append(test_finite_diff_hessian(J, np.random.randn(len(groundtruth.array)), sw))
        print('  at random %5s: %e' % ('(%d)' % (i + 1), errors[-1]))
    
    print('\nMaximum error: %e' % max(errors))
    assert max(errors) < 1e-8, 'Hessian validation failed'

print('')
print('===================================')
print('==     A L L     P A S S E D     ==')
print('===================================')


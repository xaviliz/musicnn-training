from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework.function import Defun


def get_axes(x):
    return tuple(range(len(x.shape)))

def _gradient_projection_grad(op, x, y):
    print('backward pass!')
    proj_x_y = (tf.tensordot(x, y, ((0,1), (0,1))) /
                tf.tensordot(y, y, ((0,1), (0,1)))) * y

    # return [x , y]
    return [x - proj_x_y, y]


@Defun(tf.float32, tf.float32, python_grad_func=_gradient_projection_grad)
def gradient_projection(x, y):
    return x, y

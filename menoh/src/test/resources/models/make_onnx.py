import argparse
import logging

import numpy as np
from unittest import mock

import chainer
import chainer.functions as F
import chainer.links as L

import onnx_chainer


# e.g. [[0, 0], [0, 1], [1, 0], [1, 1]] -> [[0], [0], [0], [1]]
class AND(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.fc1 = L.Linear(2, 1)

        self.fc1.W.array[:] = np.array([[1.0, 1.0]])
        self.fc1.b.array[:] = np.array([[-1.0]])

    def __call__(self, x):
        x.node._onnx_name = 'input'
        h = F.absolute(F.relu(self.fc1(x)))
        h.node._onnx_name = 'output'
        return h


# e.g. [[0, 1, 2], [3, 4, 5]] -> [[0, 0, 15, 96, 177], [0, 0, 51, 312, 573]]
class MLP(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.fc1 = L.Linear(3, 4)
            self.fc2 = L.Linear(4, 5)

        self.fc1.W.array[:] = np.arange(-6, 6).reshape((4, 3))
        self.fc1.b.array[:] = np.arange(-2, 2)

        self.fc2.W.array[:] = np.arange(-10, 10).reshape((5, 4))
        self.fc2.b.array[:] = np.arange(-2, 3)

    def __call__(self, x):
        x.node._onnx_name = 'input'
        h = F.relu(self.fc1(x))
        h.node._onnx_name = 'fc1'
        h = F.relu(self.fc2(h))
        h.node._onnx_name = 'fc2'
        return h


class IDGenerator(object):

    def __init__(self):
        # keep original
        self._id = id

    def __call__(self, obj):
        return getattr(obj, '_onnx_name', self._id(obj))


def main(logger):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--out', required=False)
    args = parser.parse_args()

    if args.out is None:
        output_filename = args.model + ".onnx"
    else:
        output_filename = args.out

    logger.info("Generating `{}` model and save it to `{}`".format(args.model, output_filename))

    try:
        if args.model == 'and_op':
            model = AND()
            x = np.empty((1, 2), dtype=np.float32)
            with chainer.using_config('train', False), \
                    mock.patch('builtins.id', IDGenerator()):
                onnx_chainer.export(model, x, filename=output_filename)

        elif args.model == 'mlp':
            model = MLP()
            x = np.empty((1, 3), dtype=np.float32)
            with chainer.using_config('train', False), \
                    mock.patch('builtins.id', IDGenerator()):
                onnx_chainer.export(model, x, filename=output_filename)

    except Exception:
        logger.exception("An error occurred during generation of the model")


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    main(logger)

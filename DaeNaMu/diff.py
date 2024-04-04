import numpy as np
from typing import List, Union, Any
import collections


class Variable:
    def __init__(self, data: np.ndarray, verbose: bool = False) -> None:
        if (data is not None) & (not isinstance(data, np.ndarray)):
            raise TypeError(f'{type(data)} is not supported.')

        self.data: np.ndarray = data
        self.grad: np.ndarray = None
        self.creator: 'Function' = None
        self.generation = 0

        self.verbose = verbose
        if verbose:
            self.vprint = self._verbose_print
        else:
            self.vprint = self._silent_print

    @staticmethod
    def _verbose_print(*inp, end: str = None) -> None:
        print(*inp, end=end)

    @staticmethod
    def _silent_print(*inp, end: str = None) -> None:
        return

    def cleargrad(self):
        self.grad = None

    def set_creator(self, func: 'Function') -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self) -> None:
        """

        """

        if self.grad is None:
            # does this count as filling in dummy data?
            self.grad = np.ones_like(self.data)

        funcs = list()
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f: 'Function' = funcs.pop()
            gys = [out.grad for out in f.outs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)


class Function:

    def __call__(self, *inputs: 'Variable') -> Any:  # Union['Variable', List['Variable']]

        self.inputs = inputs
        xs, vbs = [inp.data for inp in inputs], [inp.verbose for inp in inputs]  # decapsulate
        ys = self.forward(*xs)
        ys = ys if isinstance(ys, tuple) else (ys,)

        self.generation = max([x.generation for x in inputs])

        """
        0 dimension np arrays will return np.float after operations!
        https://stackoverflow.com/questions/77359660/why-does-operating-on-a-0d-numpy-array-give-a-numpy-float
        https://stackoverflow.com/questions/773030/why-are-0d-arrays-in-numpy-not-considered-scalar
        made to cast it back to np.array before passing it in case it is a scalar
        """
        def as_array(y):
            return np.array(y) if np.isscalar(y) else y

        outs = [Variable(as_array(y), verbose=vb) for y, vb in zip(ys, vbs)]
        [out.set_creator(self) for out in outs]
        self.outs = outs

        # checks if all the Variable's `.creator` in the `inputs` are the same
        # creators = [inp.creator.layer_id for inp in self.inputs]
        # if len(set(creators)) != 1:
        #    raise Exception('Some of the variables are from a different creator/parent!\n'
        #                    f'creator lists dump: {collections.Counter(creators)}')
        # setting layer ID if the above assertion passes
        # used creators[0]'s layer_id since it's going to be all the same anyway
        # self.layer_id = prev_layer.layer_id + 1 if (prev_layer := creators[0]) else 0

        return outs if len(outs) > 1 else outs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


# purpose of gy is a bit vague?
# gy: gradient with respect to y
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

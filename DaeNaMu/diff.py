import numpy as np
from typing import List
import collections


class Variable:
    def __init__(self, data: np.ndarray, verbose: bool = False) -> None:
        if (data is not None) & (not isinstance(data, np.ndarray)):
            raise TypeError(f'{type(data)} is not supported.')

        self.data: np.ndarray = data
        self.grad: np.ndarray = None
        self.creator: 'Function' = None

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

    def set_creator(self, func: 'Function') -> None:
        self.creator = func

    def backward(self) -> None:
        self.vprint('=== Backprop triggered! ===')
        if self.grad is None:
            # does this count as filling in dummy data?
            self.grad = np.ones_like(self.data)
            self.vprint(f'empty self.grad, creating dummy data -> {self.grad}')

        funcs: List['Function'] = [self.creator]
        i = 0
        self.vprint(f'initial funcs: {[func.layer_id for func in funcs]}')
        while funcs:
            self.vprint(f'loop iter. {i}', end=' | ')
            f: 'Function' = funcs.pop()
            self.vprint(f'popped function_id: {f.layer_id}', end=' -> ')
            x, y = f.inp, f.out
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
                self.vprint(f'funcs after appending: {[func.layer_id for func in funcs]}')
            else:
                self.vprint('breaking loop!')

            i += 1


class FunctionTemp:
    def __call__(self, inp: 'Variable') -> 'Variable':
        self.inp: 'Variable' = inp
        x = inp.data  # decapsulate
        y = self.forward(x)

        """
        0 dimension np arrays will return np.float after operations!
        https://stackoverflow.com/questions/77359660/why-does-operating-on-a-0d-numpy-array-give-a-numpy-float
        https://stackoverflow.com/questions/773030/why-are-0d-arrays-in-numpy-not-considered-scalar
        made to cast it back to np.array before passing it in case it is a scalar
        """
        y = np.array(y) if np.isscalar(y) else y

        out = Variable(y, verbose=inp.verbose)  # encapsulate / passing on the verbose value
        out.set_creator(self)
        self.out = out

        self.layer_id = prev_layer.layer_id + 1 if (prev_layer := self.inp.creator) else 0

        return out

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()


class Function:
    def __init__(self):
        self.layer_id = None

    def __call__(self, inputs: List['Variable']) -> List['Variable']:

        self.inputs: List['Variable'] = inputs
        xs, vbs = [inp.data for inp in inputs], [inp.verbose for inp in inputs]  # decapsulate
        ys = self.forward(xs)

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
        creators = [inp.creator.layer_id for inp in self.inputs]
        if len(set(creators)) != 1:
            raise Exception('Some of the variables are from a different creator/parent!\n'
                            f'creator lists dump: {collections.Counter(creators)}')
        # setting layer ID if the above assertion passes
        # used creators[0]'s layer_id since it's going to be all the same anyway
        self.layer_id = prev_layer.layer_id + 1 if (prev_layer := creators[0]) else 0

        return outs

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y, )


# purpose of gy is a bit vague?
# gy: gradient with respect to y
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inp.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inp.data
        gx = np.exp(x) * gy
        return gx

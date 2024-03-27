import numpy as np
from typing import List


class Variable:
    def __init__(self, data: np.ndarray, verbose: bool = False) -> None:
        self.data: np.ndarray = data
        self.grad: np.ndarray = np.array([None])
        self.creator = None

        self.verbose = verbose
        if verbose:
            self.vprint = self.verbose_print
        else:
            self.vprint = self.silent_print

    @staticmethod
    def verbose_print(*inp, end: str = None) -> None:
        print(*inp, end=end)

    @staticmethod
    def silent_print(*inp, end: str = None) -> None:
        return

    def set_creator(self, func: 'Function') -> None:
        self.creator = func

    def backward(self) -> None:
        funcs: List['Function'] = [self.creator]
        i = 0
        self.vprint(f'initial funcs: {[func.layer_id for func in funcs]}')
        while funcs:
            self.vprint(f'loop no.{i}', end=' | ')
            f: 'Function' = funcs.pop()
            self.vprint(f'popped function: {f.layer_id}', end=' -> ')
            x, y = f.inp, f.out
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
                self.vprint(f'funcs after appending: {[func.layer_id for func in funcs]}')
            else:
                self.vprint('breaking loop!')

            i += 1


class Function:
    def __call__(self, inp: 'Variable') -> 'Variable':
        self.inp: 'Variable' = inp
        x = inp.data  # decapsulate
        y = self.forward(x)

        out = Variable(y, verbose=inp.verbose)  # encapsulate / passing on the verbose value
        out.set_creator(self)
        self.out = out

        self.layer_id = prev_layer.layer_id + 1 if (prev_layer := self.inp.creator) else 0

        return out

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()


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

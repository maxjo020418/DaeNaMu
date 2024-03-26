import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data
        self.grad: np.ndarray = np.array([None])


class Function:
    def __call__(self, inp: Variable) -> Variable:
        self.inp: Variable = inp
        x = inp.data  # decapsulate
        y = self.forward(x)
        output = Variable(y)  # encapsulate
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()


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

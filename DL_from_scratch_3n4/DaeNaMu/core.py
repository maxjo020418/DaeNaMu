import numpy as np
import weakref
from typing import Any, List
from dataclasses import dataclass
# from memory_profiler import profile


@dataclass
class Config:
    verbose: bool = True
    enable_backprop: bool = True


def as_array(y):
    return np.array(y) if np.isscalar(y) else y


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Variable:
    """
    might be able to change this into a @dataclass?
    """

    def __init__(self, data: np.ndarray, name: str = 'default') -> None:
        if (data is not None) & (not isinstance(data, np.ndarray)):
            raise TypeError(f'{type(data)} is not supported.')

        self.data: np.ndarray = data
        self.grad: np.ndarray = None
        self.creator: 'Function' = None
        self.generation = 0

        self.verbose = Config.verbose
        if self.verbose:
            self.vprint = self._verbose_print
        else:
            self.vprint = self._silent_print

        self.name_suffix = name
        self.name = self.__class__.__name__ + '_' + str(self.generation)
        if name != 'default':
            self.name = self.name + '_' + self.name_suffix

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:  # when None
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return f'Variable({p})'

    # setting the priority to not get overwritten/casted by other classes
    # eg: `Variable + np.array = Variable` will always happen if
    # __array_priority__ is set to 200 (higher than np.array)
    __array_priority__ = 200

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return self.sub(other)

    def __mul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        return self.rdiv(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __pow__(self, power):
        return self.pow(power)

    def add(self, other):
        other = as_array(other)
        return Add()(self, other)

    def sub(self, other):
        other = as_array(other)
        return Sub()(self, other)

    def mul(self, other):
        other = as_array(other)
        return Mul()(self, other)

    def div(self, other):
        other = as_array(other)
        return Div()(self, other)

    def rdiv(self, other):
        other = as_array(other)
        return Div()(other, self)

    def pow(self, c):
        return Pow(c)(self)

    def neg(self):
        return Neg()(self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.size

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def cleargrad(self):
        self.grad = None

    @staticmethod
    def _verbose_print(*inp, end: str = None) -> None:
        print(*inp, end=end)

    @staticmethod
    def _silent_print(*inp, end: str = None) -> None:
        return

    def set_creator(self, func: 'Function') -> None:
        self.creator = func
        self.generation = func.generation + 1

        self.name = self.__class__.__name__ + '_' + str(self.generation)
        if self.name_suffix != 'default':
            self.name = self.name + '_' + self.name_suffix

    # @profile
    def backward(self, retain_grad=False, create_graph=False) -> None:
        """

        """
        self.vprint('=== begin backprop ===')

        if self.grad is None:
            """
            it's common in such frameworks to default this initial gradient to 1. 
            This means it assumes the gradient of the loss with respect to y is 1, 
            essentially simulating a loss function where the derivative of the loss with respect to its input is 1.
            """
            # self.grad = np.ones_like(self.data) # for the old method
            self.grad = Variable(np.ones_like(self.data))
            self.vprint(f'filled in default grad {self.grad}')

        funcs = list()
        seen_set = set()

        def add_func(func):
            # if statement made in case there are multiple inputs (at branch start)
            if func not in seen_set:
                self.vprint(f'added {func.name} to `funcs` and `seen_set`')
                funcs.append(func)
                seen_set.add(func)
                funcs.sort(key=lambda inp: inp.generation)
            else:
                self.vprint(f'{func.name} in `seen_set`!')

        add_func(self.creator)
        self.vprint('loop init:\n', 'funcs:', [func.name for func in funcs],
                    '\nseen_set:', [seen.name for seen in seen_set], '\n', '='*10)

        i = 1
        while funcs:
            self.vprint('loop no.', i)
            f: 'Function' = funcs.pop()
            self.vprint(f'popped {f.name}, remaining funcs: {[func.name for func in funcs]}')

            # outputs is a list of weakrefs
            # https://chat.openai.com/share/136a9879-bd08-40f7-a9f8-97bd02821012
            gys = [out().grad for out in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                self.vprint(f'gys: {gys}, gxs: {gxs}')

                if not isinstance(gxs, tuple):
                    self.vprint('converting gxs to tuple...')
                    gxs = (gxs,)

                self.vprint(f'f.inputs: {[var.name for var in f.inputs]} | '
                            f'f.outputs (weakrefs): {[var().name for var in f.outputs]}')
                for x, gx in zip(f.inputs, gxs):
                    # if statement for ADD operations (p.123)
                    if x.grad is None:
                        x.grad = gx
                    else:
                        self.vprint(f'gradient added! -> {x.grad} + {gx}')
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)
                    else:
                        self.vprint('breaking loop!')


            if not retain_grad:
                self.vprint(f'deleting grads: {[var().name for var in f.outputs]}')
                for y in f.outputs:
                    y().grad = None
                    # delattr(y(), 'grad')

            self.vprint('='*10)
            i += 1


class Function:
    def __call__(self, *inputs: 'Variable', name: str = 'default') -> Any:

        inputs = [as_variable(x) for x in inputs]

        xs = [inp.data for inp in inputs]  # decapsulate
        ys = self.forward(*xs)
        ys = ys if isinstance(ys, tuple) else (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        # 원래 아래 2줄은 아래 if statement 안에 있어야 하는데, 이름 표기 문제 이유 때문에 밖에 둠.
        self.generation = max([x.generation for x in inputs])
        [out.set_creator(self) for out in outputs]

        if Config.enable_backprop:
            self.inputs = inputs
            # self 있어야 함! 안 그러면 reference count 날라감!
            self.outputs = [weakref.ref(out) for out in outputs]  # p.149

        self.name_suffix = name
        self.name = self.__class__.__name__ + '_' + str(self.generation)
        if name != 'default':
            self.name = self.name + '_' + self.name_suffix

        """
        0 dimension np arrays will return np.float after operations!
        https://stackoverflow.com/questions/77359660/why-does-operating-on-a-0d-numpy-array-give-a-numpy-float
        https://stackoverflow.com/questions/773030/why-are-0d-arrays-in-numpy-not-considered-scalar
        made to cast it back to np.array before passing it in case it is a scalar
        """

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()

# =================================================================================== #

# import numpy as np
# from .main import Function, Config
import contextlib


@contextlib.contextmanager
def using_config(name: str, value):
    old_val = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_val)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -1 * gy


# gy: y's gradient with respect to output(can be Loss)
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        # using the first element of `inputs`,
        # every element inside `inputs` should be all same.
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


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = self.c * x ** (self.c - 1) * gy
        return gx


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


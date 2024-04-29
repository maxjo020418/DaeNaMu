import numpy as np
import weakref
from typing import Any, List
from dataclasses import dataclass


@dataclass
class Config:
    verbose: bool = True


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

        self.name = self.__class__.__name__ + '_' + str(self.generation)
        if self.name_suffix != 'default':
            self.name = self.name + '_' + self.name_suffix

    def backward(self, retain_grad=False) -> None:
        """

        """
        self.vprint('=== begin backprop ===')

        if self.grad is None:
            # I think this counts as filling in dummy data
            self.grad = np.ones_like(self.data)
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
        self.vprint('loop init:\n', 'funcs:', [func.name for func in funcs], '\nseen_set:', [seen.name for seen in seen_set], '\n', '='*10)

        i = 1
        while funcs:
            self.vprint('loop no.', i)
            f: 'Function' = funcs.pop()
            self.vprint(f'popped {f.name}, remaining funcs: {[func.name for func in funcs]}')

            # outs is a list of weakrefs
            # https://chat.openai.com/share/136a9879-bd08-40f7-a9f8-97bd02821012
            gys = [out().grad for out in f.outs]

            gxs = f.backward(*gys)
            self.vprint(f'gys: {gys}, gxs: {gxs}')

            if not isinstance(gxs, tuple):
                self.vprint('converting gxs to tuple...')
                gxs = (gxs,)

            self.vprint(f'f.inputs: {[var.name for var in f.inputs]}')
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

            self.vprint('='*10)
            i += 1


class Function:
    def __call__(self, *inputs: 'Variable', name: str = 'default') -> Any:

        self.inputs = inputs
        xs = [inp.data for inp in inputs]  # decapsulate
        ys = self.forward(*xs)
        ys = ys if isinstance(ys, tuple) else (ys,)

        self.generation = max([x.generation for x in inputs])

        self.name_suffix = name
        self.name = self.__class__.__name__ + '_' + str(self.generation)
        if name != 'default':
            self.name = self.name + '_' + self.name_suffix

        """
        0 dimension np arrays will return np.float after operations for whatever reason!!
        https://stackoverflow.com/questions/77359660/why-does-operating-on-a-0d-numpy-array-give-a-numpy-float
        https://stackoverflow.com/questions/773030/why-are-0d-arrays-in-numpy-not-considered-scalar
        made to cast it back to np.array before passing it in case it is a scalar
        """

        def as_array(y):
            return np.array(y) if np.isscalar(y) else y

        outs = [Variable(as_array(y)) for y in ys]
        [out.set_creator(self) for out in outs]
        self.outs = [weakref.ref(out) for out in outs]  # p.149

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

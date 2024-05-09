from DaeNaMu.main import *


def _dot_var(v: 'Variable', verbose=False):
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += f'{v.shape} {v.dtype}'

    return f'{id(v)} [label="{name}", color=orange, style=filled]\n'


def _dot_func(f: 'Function'):
    txt = f'{id(f)} [label="{f.name}"]\n'

    for x in f.inputs:
        txt += f'{id(x)} -> {id(f)}\n'
    for y in f.outs:
        txt += f'{id(f)} -> {id(y())}\n'

    return txt


def get_dot_graph(output: 'Variable', verbose=False):
    txt = str()
    funcs = []
    seen_set = set()

    def add_func(f: 'Function'):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return f'digraph g {{\n{txt}}}'

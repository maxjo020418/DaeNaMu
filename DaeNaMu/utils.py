from DaeNaMu import *
import os
import subprocess


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
    for y in f.outputs:
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


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # dot 데이터를 파일에 저장
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir): #~/.dezero 디렉터리가 없다면 새로 생성
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    #dot 명령 호출
    extension = os.path.splitext(to_file) [1] [1:] # png, pdf...
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

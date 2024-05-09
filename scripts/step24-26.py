from DaeNaMu import *


def goldstein(x0, x1):
    return \
        ((1 + (x0 + x1 + 1) ** 2 * (19 - 14 * x0 + 3 * x0 ** 2 - 14 * x1 + 6 * x0 * x1 + 3 * x1 ** 2)) *
         (30 + (2 * x0 - 3 * x1) ** 2 * (18 - 32 * x0 + 12 * x0 ** 2 + 48 * x1 - 36 * x0 * x1 + 27 * x1 ** 2)))


def sphere(x0, x1):
    return x0**2 + x1**2


def rosenbrock(x0, x1):
    return 100 * (x1 - x0**2)**2 + (1 - x0)**2


x0 = Variable(np.array(0.0), name='x0')
x1 = Variable(np.array(2.0), name='x1')
y = rosenbrock(x0, x1)
y.backward()
print(x0.grad, x1.grad)

with open(f'../outputs/rosenbrock-graph.dot', 'w') as f:
    f.write(get_dot_graph(y))

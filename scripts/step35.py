import numpy as np
from DaeNaMu import *

x = Variable(np.array(1.0))
y = Tanh()(x)
y.backward(create_graph=True)

iters = 4

for i in range(iters):
    print(f'LOOP => {i}')
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = 'gx' + str(iters+1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')

print('done:', iters + 1)

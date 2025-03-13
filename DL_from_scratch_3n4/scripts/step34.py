import numpy as np
from DaeNaMu import *

x = Variable(np.array(1.0))
y = Sin()(x)
y.backward(create_graph=True)

for i in range(3):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    print(f'x.grad: {x.grad}')

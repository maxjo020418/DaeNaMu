import numpy as np

from DaeNaMu import *

x = Variable(np.array(3.0))
y = Add()(x, x)
y = Add()(y, x)

print('y: ', y.data)
y.backward()

print('x grad: ', x.grad)
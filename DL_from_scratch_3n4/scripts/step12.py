import numpy as np

from DaeNaMu import *

x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = Add()(Square()(x), Square()(y))
z.backward()

print(z.data)
print(x.data)
print(y.data)

import numpy as np
from DaeNaMu import *

x = Variable(np.array([[1,2,3], [4,5,6]]))
c = Variable(np.array([[10,20,30], [40,50,60]]))
y = x + c

y.backward(retain_grad=True)

print(y)

print()
print(y.grad)
print(c.grad)
print(x.grad)

import numpy as np
from DaeNaMu import *

x = Variable(np.array([[1,2,3], [4,5,6]]))
y = Reshape((6,))(x)
y.backward(retain_grad=True)

print(x.grad)

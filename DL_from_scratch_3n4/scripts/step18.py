from DaeNaMu import *

with using_config('verbose', False):
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = Add()(x0, x1)
    y = Add()(x0, t)
    y.backward()

print(y.data)
print(x0.grad, x1.grad)

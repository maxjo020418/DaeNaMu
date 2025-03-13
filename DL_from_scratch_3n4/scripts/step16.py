from DaeNaMu import *
"""
for i in range(10):
    x = Variable(np.random.randn(10000))
    y = Square()(Square()(Square()(x)))
"""

x = Variable(np.array(2.0), name='x')
a = Square()(x, name='a')
y = Add()(
    Square()(a, name='y-left'),
    Square()(a, name='y-right'))
y.backward()

print(y)
print(x.grad)

with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = Square()(x)
    print(y)

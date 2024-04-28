from DaeNaMu import *
"""
for i in range(10):
    x = Variable(np.random.randn(10000))
    y = Square()(Square()(Square()(x)))
"""

x = Variable(np.array(2.0))
a = Square()(x)
y = Add()(Square()(a), Square()(a))
y.backward()

print(y.data)
print(x.grad)

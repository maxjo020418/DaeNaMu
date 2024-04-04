from DaeNaMu import *

x = Variable(np.array(2.0))
a = Square()(x)
y = Add()(Square()(a), Square()(a))
y.backward()

print(y.data)
print(x.grad)

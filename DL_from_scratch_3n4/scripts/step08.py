from DaeNaMu import *

# forward prop
x = Variable(np.array(0.5), verbose=True)
a = Square()(x)     # 0
b = Exp()(a)        # 1
y = Square()(b)     # 2

# backprop
# y.grad = np.array(1.0)
y.backward()
print(x.grad)

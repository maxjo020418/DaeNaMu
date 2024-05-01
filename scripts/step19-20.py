from DaeNaMu import *

printtest = Variable(np.array([[1, 2], [3, 4]]))
print(printtest)
print()

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = a * b + c
y.backward()

print(y)
print(a.grad)
print(b.grad)

import unittest
from DaeNaMu import *


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = Square()(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = Square()(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)


if __name__ == '__main__':
    unittest.main()

import DaeNaMu
import numpy as np

dnm = DaeNaMu.diff
f = dnm.Square()

v1 = dnm.Variable(np.array([1, 2, 3, 4]))
print(f(v1).data)

print(dnm.Variable(np.array([])).grad)

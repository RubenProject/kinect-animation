import numpy as np
from math import pi

def gen_phase_labels(x):
    _frame_count = len(x)
    #interpolate
    i = 0
    while i < _frame_count:
        if x[i] != -1.0 and x[i] != pi * 2:
            j = i + 1
            while j < _frame_count and x[i] > x[j]:
                j += 1
            print i, j
            if j == _frame_count:
                x[_frame_count - 1] = pi * 2
                j = _frame_count - 1
            for k in range(i + 1, j):
                x[k] = x[i] + (x[j] - x[i]) / (j - i) * (k - i)
            i = j - 1
        i += 1
    #get rid of any leading or trailing -1
    for i in range(_frame_count):
        if x[i] == -1:
            x[i] = 0.0
    return x



x = [-1.0] * 30
x[4] = 0.0
x[5] = pi
x[15] = pi * 2
x[16] = 0.0
x[25] = pi
x[28] = pi * 2
y = np.array(x)
print y
print(gen_phase_labels(y))

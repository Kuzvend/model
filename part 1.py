from math import sin, sqrt
from numpy import arange, vectorize, abs
from matplotlib import pyplot as plt
import csv
import os

A = 418.9829
x_start = -500.0
x_end = 500.0
x_step = 0.01


def f(x):
    return A - float(x) * sin(sqrt(abs(x)))


x_arrays = arange(x_start, x_end, x_step)
f2 = vectorize(f)
y_arrays = f2(x_arrays)

plt.plot(x_arrays, y_arrays)
plt.show()

if not os.path.isdir('result'):
    os.mkdir('result')
else:
    print('Уже есть такая директрория')

counter = 0
with open('result/data.csv', 'w') as file:
    writer = csv.writer(file)
    for x, y in zip(x_arrays, y_arrays):
        counter += 1
        writer.writerow([counter, x, y])


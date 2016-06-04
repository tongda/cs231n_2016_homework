activate_this = '.env/bin/activate_this.py'
#execfile(activate_this, dict(__file__=activate_this))
exec(open(activate_this).read(), dict(__file__=activate_this))

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

#dists = np.random((100, 100), 350, dtype=np.int64)
dists = np.random.rand(500, 700) * 255

# Compute the x and y coordinates for points on sine and cosine curves
# x = np.arange(0, 3 * np.pi, 0.1)
# y_sin = np.sin(x)
# y_cos = np.cos(x)

# Plot the points using matplotlib
# plt.plot(x, y_sin)
# plt.plot(x, y_cos)
# plt.xlabel('x axis label')
# plt.ylabel('y axis label')
#plt.title('Sine and Cosine')
# plt.legend(['Sine', 'Cosine'])
plt.imshow(dists, interpolation='none')

#plt.show()

x = np.arange(3072)
#print()
print(x.shape)
x = np.reshape(x, (x.size, 1))
print(x.shape)

y = np.ones((1, 5000))
print(y.shape)

xExtend = x * y
print(xExtend.shape)

a = np.array([[1,2], [3,4]])
b = np.array([5,0])

print(a[1])


scores = np.array([
    [-0.19,  0.039,  0.28,  0.28,  0.35, -0.01000, 0.00,  0.03,  0.08,  -0.11],
    [ 0.46, -0.39, -0.29,  0.13,  0.08, -0.21,  0.370000,  0.11,  0.017, -0.05,]
])

y = np.array([
    5, 6
])

print(scores.T[y].T)
#print(np.diag(scores.T[y]))
#np.diag(scores.T[y]) = 0
#print(socres)

#xExtend - 
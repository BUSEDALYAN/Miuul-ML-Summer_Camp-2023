#### DATA ANALYSIS WITH PYTHON

#### NUMPY
    # - Numerical Python

import numpy as np
a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []


##pyhton way 1:
for i in range(0, len(a)):
    ab.append(a[i] * b[i])

##numpy way 2:
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])

a * b

## Creating Numpy Arrays

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))

np.zeros(10,  dtype=int)
np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3, 4))

## Attributes of Numpy Arrays

a = np.random.randint(10, size =5)
a.ndim # number of dimension
a.shape # size
a.size # total element size
a.dtype # type

## Reshaping

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

## Index Selection

a = np.random.randint(10, size=10)
a[0:5]
a[0] = 99

m = np.random.randint(10, size=(3, 5))

m[0,0] #first row, then column
m[2 ,3] = 2.9 #it shows a float number as an integer

m[:,0] #all rows, 1.col
m[0:2, 0:3]

## Fancy index

v = np.arange(0, 30, 3)
v[1]

catch = [1, 2, 3]
v[catch]

## Conditions on NumPy

v = np.array([1, 2, 3, 4, 5])


###python way 1:

ab =[]

for i in v:
    if i < 3:
        ab.append(i)

###numpy way 2:

v < 3

v[v < 3]
v[v != 3]

## Mathematical Operations

v = np.array([1, 2, 3, 4, 5])

v / 5
v ** 2

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.max(v)
np.min(v)
np.var(v)

##### Equation solution with two unknowns

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array(([12, 10]))

np.linalg.solve(a, b)

###############
a = np.array([2, 4, 6, 10])

a**2

np.arange(1,7).reshape(3, 2)
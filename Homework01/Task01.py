import numpy as np

# 1
V = np.zeros(100)
print(V)

# 2
V = np.ones(100)
print(V)

# 3
V = np.full(100, 3)
print(V)

# 4
python3 -c "import numpy; numpy.info(numpy.add)"

# 5
A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)

# 6
V = np.arange(10,101)
print(V)

# 7
V = np.arange(10)
V = V[::-1]
print(V)

# 8
A = np.arange(1,17).reshape(4,4)
print(A)

# 9
V = np.array([1,2,3,4,5])
n = 3
Vn = np.zeros(len(V) + (len(V) - 1) * (n))
Vn[::n + 1] = V
print(Vn)

# 10
V = np.eye(3)
print(V)

# 11
V = np.random.random((3,3,3))
print(V)

# 12
V = np.random.random((10,10))
Vmin, Vmax = V.min(), V.max()
print(Vmin,Vmax)

# 13
V = np.random.random(30)
m = V.mean()
print(m)

# 14
V = np.ones((10,10))
V[1:-1,1:-1] = 0
print(V)

# 15 (транспонирование матрицы)
A = np.arange(1,17).reshape(4,4)
A = A.transpose()
print(A)

# 16
A = np.diag(np.arange(1, 5), k=0)
print(A)

# 17
A = np.zeros((8,8), dtype=int)
A[1::2,::2] = 1
A[::2,1::2] = 1
print()

# 18
print(np.unravel_index(100, (6,7,8)))

# 19
A = np.tile(np.array([[0,1],[1,0]]), (4,4))
print(A)

# 20
A = np.dot(np.ones((5,3)), np.ones((3,2)))
print(A)

# 21
A = np.arange(11)
A[(A > 3) & (A < 8)] *= -1
print(A)

# 23
A = np.zeros((5,5))
A += np.arange(5)
print(A)

# 24
def generator():
    for x in range(10):
        yield x
A = np.fromiter(generator(),dtype=int,count=-1)
print(A)

# 25
A = np.linspace(0,1,12)[1:-1]
print(A)

# 26
A = np.random.random(5)
A.sort()
print(A)

# 27
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
print(A,B,np.allclose(A,B))

# 28
A = np.zeros(10)
A.flags.writeable = False
# A[0] = 1 throws ValueError: assigment destination is read-only

# 29
A = np.arange(10000)
np.random.shuffle(A)
n = 5
print (A[np.argpartition(-A,n)[:n]])

# 30
V = np.random.random(10)
V[V.argmax()] = 0
print(V)

# 31
A = np.zeros((10,10), [('x',float),('y',float)])
A['x'], A['y'] = np.meshgrid(np.linspace(0,1,10),
                             np.linspace(0,1,10))
print(A)

# 32
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))

# 33
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)

# 34
np.set_printoptions(threshold=np.nan)
A = np.arange(1,26).reshape((5,5))
print(A)

# 35
V = np.arange(100)
V1 = np.random.uniform(0,100)
index = (np.abs(V - V1)).argmin()
print(V[index])

# 36
V = np.arange(10, dtype=np.int32)
print(V)
V = V.astype(np.float32, copy=False)
print(V)

# 37
A = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(A):
    print(index, value)
for index in np.ndindex(A.shape):
    print(index, A[index])

# 38
p = 3
n = 10
A = np.zeros((n,n))
np.put(A, np.random.choice(range(n*n), p, replace=False), 1)
print(A)

# 39
X = np.random.rand(5, 10)
Y = X - X.mean(axis=1, keepdims=True)
print(X)

# 40
A = np.random.randint(0,10,(3,3))
n = 1
print(A)
print(A[A[:,n].argsort()])

# 41
A = np.random.randint(0,3,(3,10))
print(A)
print((~A.any(axis=0)).any())

# 42
A = np.ones(10)
B = np.random.randint(0,len(A),20)
A += np.bincount(B, minlength=len(A))
print(A)

# 43
w = 16
h = 16
A = np.random.randint(0, 2, (w,h,3)).astype(np.ubyte)
B = A[...,0] * 256 * 256 + A[...,1] * 256 + A[...,2]
n = len(np.unique(B))
print(np.unique(A))

# 44
A = np.random.randint(0,10, (3,4,3,4))
s = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(s)
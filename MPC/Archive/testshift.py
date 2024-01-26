import numpy as np

# np.random.choice(range(1,5)) <-- [1-4]

def shiftRows(arr_2d,r):
    return np.roll(arr_2d, r, axis=0)

def shiftCols(arr_2d,c):
    return np.roll(arr_2d, c, axis=1)
    

arr = np.array([[0,0,0,0],
               [1,0,0,0],
               [1,1,0,0],
               [1,1,1,0]])

arr = shiftRows(arr,3)
arr = shiftCols(arr,2)
print(arr)


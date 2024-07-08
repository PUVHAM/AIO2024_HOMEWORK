import numpy as np

# Question 1: Which of the following is correct to create a 1-dimensional array from 0 to 9
arr = np.arange(0, 10, 1)
print(arr)

# Question 2: How to create a 3x3 boolean array with all values as True
arr = np.full((3, 3), fill_value=True, dtype=bool)
print(arr)

# Question 3: Result of the following code:
arr = np.arange(0, 10)
print(arr[arr % 2 == 1]) # [1 3 5 7 9]

# Question 4: Result of the following code:
arr = np.arange(0, 10)
arr[arr % 2 == 1] = -1
print(arr) # [0 -1 2 -1 4 -1 6 -1 8 -1]

# Question 5: Result of the following code:
arr = np.arange(10)
arr_2d = arr.reshape(2, -1)
print(arr_2d) # [[0 1 2 3 4]
              #  [5 6 7 8 9]]

# Question 6: Result of the following code:
arr1 = np.arange(10).reshape(2, -1)
arr2 = np.repeat(1, 10).reshape(2, -1)
c = np.concatenate([arr1, arr2], axis=0)
print("Result: \n", c)
# Result:
# [[0 1 2 3 4]
#  [5 6 7 8 9]
#  [1 1 1 1 1]
#  [1 1 1 1 1]]

# Question 7: Result of the following code:
arr1 = np.arange(10).reshape(2, -1)
arr2 = np.repeat(1, 10).reshape(2, -1)
c = np.concatenate([arr1, arr2], axis=1)
print("C = ", c)
# C = [[0 1 2 3 4 1 1 1 1 1]
#      [5 6 7 8 9 1 1 1 1 1]]

# Question 8: Result of the following code:
arr = np.array([1, 2, 3])
print(np.repeat(arr, 3))
print(np.tile(arr, 3))
# [1 1 1 2 2 2 3 3 3]
# [1 2 3 1 2 3 1 2 3]

# Question 9: Result of the following code:
a = np.array([2, 6, 1, 9, 10, 3, 27])
index = np.nonzero((a>=5)&(a<=10))
print("result", a[index]) # result [6 9 10]

# Question 10: Result of the following code:
def maxx(x, y):
    if x >= y:
        return x
    else:
        return y
    
a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 2])

pair_max = np.vectorize(max, otypes=[float])
print(pair_max(a, b)) # [6. 7. 9. 8. 9. 7. 5.]

# Question 11: Result of the following code:
a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])

print("Result", np.where(a < b, b, a)) #[6 7 9 8 9 7 5]

def factorial(num):
    factorial = 1
    for i in range(1, num + 1):
        factorial *= i
    return factorial

#Calculate the sine of x
def approx_sin(x, n):
    assert isinstance(n, int) and n > 0
    
    result = 0
    for num in range(n):
        result += ((-1)**num)*(x**(2*num + 1)/factorial(2*num + 1))
    return result

#Calculate the cosine of x
def approx_cos(x, n):
    assert isinstance(n, int) and n > 0

    result = 0
    for num in range(n):
        result += ((-1)**num)*(x**(2*num)/factorial(2*num))
    return result

def approx_cos(x, n):
    assert isinstance(n, int) and n > 0

    result = 0
    for num in range(n):
        result += ((-1)**num)*(x**(2*num)/factorial(2*num))
    return result

def approx_sinh(x, n):
    assert isinstance(n, int) and n > 0

    result = 0
    for num in range(n):
        result += (x**(2*num + 1)/factorial(2*num + 1))
    return result

def approx_cosh(x, n):
    assert isinstance(n, int) and n > 0

    result = 0
    for num in range(n):
        result += (x**(2*num)/factorial(2*num))
    return result

#Testcases
print(approx_sin(x=3.14, n=10))
print(approx_cos(x=3.14, n=10))
print(approx_sinh(x=3.14, n=10))
print(approx_cosh(x=3.14, n=10))








import math

def md_nre_single_sample(y, y_hat, n, p):
    return (y**(1/n) - y_hat**(1/n))**p

#Testcases
print(md_nre_single_sample(y = 100, y_hat = 99.5, n = 2, p = 1))
print(md_nre_single_sample(y = 20, y_hat = 19.5, n = 2, p = 1))
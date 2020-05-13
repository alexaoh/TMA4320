#Composite Midpoint method

def midpoint_method(f, a, b, n):
    ''' 
    Input: 
    f: Function to integrate
    a: Beginning point
    b: End point
    n: Number of intervals. 

    Output: 
    Numerical approximation to the integral
    '''
    assert(b > a)
    h = (b-a)/n
    result = 0
    for i in range(1,n+1):
        result += f(a+h*i-(h/2))
    return h*result

#Implementation with numpy (no loop):

import numpy as np

def midpoint_numpy(f, a, b, n):
    ''' 
    Input: 
    f: Function to integrate
    a: Beginning point
    b: End point
    n: Number of intervals. 

    Output: 
    Numerical approximation to the integral
    '''
    assert(b > a)
    h = (b-a)/n
    x = np.linspace(a,b,n+1)
    return h*np.sum(f(x[1::]-(h/2)))


#Test
print(midpoint_method(lambda x : np.exp(-x**2), 0, 2, 16))
print(midpoint_numpy(lambda x : np.exp(-x**2), 0, 2, 16))
#They give (approximately) the same result!

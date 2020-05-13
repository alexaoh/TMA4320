#Composite trapezoid quadrature implementation

def trapezoid(f,a,b,n):
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
    h = float(b-a)/n
    result = (f(a) + f(b))/2.0
    for k in range(1,n):
        result += f(a + k*h)
    return h*result

#Implementation with numpy (no loop):

import numpy as np

def trapezoid_numpy(f, a, b, n):
    assert(b > a)
    h = (b-a)/n
    x = np.linspace(a,b,n+1)
    result = (f(x[0]) + f(x[-1]))*(h/2)
    return result + h*np.sum(f(x[1:-1:]))

#Test
print(trapezoid(lambda x : np.exp(-x**2), 0, 2, 16))
print(trapezoid_numpy(lambda x : np.exp(-x**2), 0, 2, 16))
#They give the same result!

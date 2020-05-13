#Composite Simpson's quadrature

def simpsons_method(f, a, b, n):
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
    assert(n % 2 == 0) #n must be even. 
    h = float(b-a)/n
    m = int(n/2) 
    result = f(a) + f(b)
    for i in range(1,m): #Even terms
        result += 2*f(a+2*i*h)
    for i in range(1,m+1): #Odd terms
        result += 4*f(a+(2*i-1)*h)
    return (h/3)*result

import numpy as np

#Test
print(simpsons_method(lambda x : 3*x**2, 0, 1, 10))
print(simpsons_method(np.sin, 0, np.pi/2, 100))


#Implementation with numpy (no loops):

def simpsons_numpy(f, a, b, n):
    ''' 
    Input: 
    f: Function to integrate
    a: Beginning point
    b: End point
    n: Number of intervals. 

    Output: 
    Numerical approximation to the integral
    '''
    assert(n % 2 == 0)
    m = n/2
    x = np.linspace(a,b,n+1) 
    h = (b-a)/n
    s1 = f(x[0]) + f(x[-1]) #Starting point and endpoint
    s2 = 2*np.sum(f(x[2:-1:2])) #Even points
    s3 = 4*np.sum(f(x[1::2])) #Odd points
    return (h/3)*(s1+s2+s3)

#Test
print(simpsons_numpy(lambda x : 3*x**2, 0, 1, 10)) #Do not know why this is different? Overflow somewhere?
print(simpsons_numpy(np.sin, 0, np.pi/2, 100))

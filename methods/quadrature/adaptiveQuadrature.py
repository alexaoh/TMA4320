#Adaptive quadrature with the trapezoid method

import numpy as np
def trapezoid_step(a, b, f):
    ''' Trapezoid rule in one interval ''' 
    return (f(a)+f(b))*(b-a)/2

def adaptive_quadrature(f, a, b, tol):
    ''' 
    Input: 
    f: Function to integrate
    a: Beginning point
    b: End point
    tol: Error tolerance

    Output: 
    Numerical approximation to the integral
    '''
    #Have not gotten this to work as planned. 
    n = 1
    f_call_counter = 0

    points = np.zeros((2,2)) #Holds a's in first column and b's in second column
    points[0,:] = (a,b)

    approx = np.zeros(1)
    approx[0] = trapezoid_step(a, b, f)
    f_call_counter += 2

    tolerance = np.zeros(1)
    tolerance[0] = tol

    result = 0
    while n > 0:
        c = (points[n-1,0]+points[n-1,1])/2
        approx = np.append(approx, trapezoid_step(points[n-1,0], points[n-1,1], f))
        f_call_counter += 2
        approx = np.append(approx, trapezoid_step(c, points[n-1,1], f))
        f_call_counter += 2
        #print(tolerance[n-1])
        #print(np.abs(approx[n]-approx[n+1]-approx[n-1]))
        if np.abs(approx[n]-approx[n+1]-approx[n-1]) < 3*tolerance[n-1]:
            print("heisann")
            result += approx[n] + approx[n+1]
            n -= 1
        else: 
            points = np.vstack((points, [c, points[n-1,1]]))
            points[n-1,1] = c
            #print(tolerance[n-1]/2)
            tolerance = np.append(tolerance, tolerance[n-1]/2)
            #tolerance[n-1] /= 2
            n += 1
            print(n)
    return result, f_call_counter

#Test
#result, counter = adaptive_quadrature(lambda x : np.sin(x**2), 0, 5, 1.0e-7)
#print(counter)

#Recursive implementation of the adaptive quadrature with the trapezoid method

def recursive_adaptive_quadrature(f, a, b, tol, old_approx):
    ''' 
    Input: 
    f: Function to integrate
    a: Beginning point
    b: End point
    tol: Tolerance of error in approximation in the interval. 
    old_approximation: Used to calculate the approximation to the correct tolerance level.  

    Output: 
    Numerical approximation to the integral
    '''
    c = (a+b)/2
    Sac = trapezoid_step(a, c, f)
    Scb = trapezoid_step(c, b, f)
    global function_calls #Used to count number of function calls (easiest, but not great solution). 
    function_calls += 4
    new_approx = Sac + Scb
    if np.abs(new_approx-old_approx) < 3*tol:
        return new_approx
    else: 
        tol /= 2
        return recursive_adaptive_quadrature(f, a, c, tol, Sac) +\
               recursive_adaptive_quadrature(f, c, b, tol, Scb)

#Test recursive_adaptive_quadrature. 
a = 0
b = 1
tol = 1.0e-8
function_calls = 2
old_approximation = trapezoid_step(a, b, lambda x : np.exp(x)/np.cos(x))
approximation = recursive_adaptive_quadrature(lambda x : np.exp(x)/np.cos(x), a, b, tol, old_approximation)
print(approximation)
print(function_calls)
# Fixed point iteration. The same function is used with Newton's method, 
# with g(x) = x - f(x)/f'(x) (f(x)=0) as input instead of regular g(x) = x iteration. 


def fixit(x0,tol, g):
    assert tol>0
    maxiter = 100 
    iter = 0
    errest = 2*tol
    while errest > tol and iter < maxiter:
        iter += 1
        x = g(x0)
        errest = abs(x-x0)
        x0 = x
    return x, iter

#Systems of equations with Newton's method:
#The same iteration as above will work, but need to compute a Jacobian matrix
#and solve a linear system of equations. This can be done with some of the 
#methods in the linearAlgebra-directory, depending in which will converge. 
#Below, the system is solved with numpy-linalg, and an example of Jacobian computation
#by automatic differentiation is shown:


import autograd.numpy as np #Regular numpy does not work with automatic differentiation.
from autograd import jacobian
import numpy.linalg as la

def sNewton(f,x0,tol):
    assert tol>0
    maxiter = 100 
    iter = 0
    errest = 2*tol #To ensure that the loop starts. 
    while errest > tol and iter < maxiter:
        iter += 1
        Df = jacobian(f)
        A = Df(x0)
        b = -f(x0)
        delta = la.solve(A,b)
        x = x0 + delta
        errest = la.norm(delta)
        x0 = x
    return x, iterb

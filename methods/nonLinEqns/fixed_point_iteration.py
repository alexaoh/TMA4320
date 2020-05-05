# Fixed point iteration. The same function is used with Newton's method, 
# with g(x) = x - f(x)/f'(x) (f(x)=0) as input instead of regular g(x) = x iteration. 


def fixit(x0,tol, g):
    assert tol>0
    maxiter = 100 #for example
    iter = 0
    errest = 2*tol
    while errest > tol and iter < maxiter:
        iter += 1
        x = g(x0)
        errest = abs(x-x0)
        x0 = x
    return x, iter

#Have a look at Bonus also, when implemented all of this (from Øving 2)

#Systems of equations coming up next:
#The same function fixit() should work?

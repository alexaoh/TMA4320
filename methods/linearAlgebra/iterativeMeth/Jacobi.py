import numpy as np 

def jacobi(A, b, u0, tol, maxiter):
    k = 0
    est = 2*tol
    D = np.diag(np.diag(A))
    D_inverse = np.diag(1/np.diag(A))
    N = D-A
    while est > tol:
        print("Iteration ",k,u0)
        k += 1   

        u = D_inverse@(N@u0+b)

        est = np.linalg.norm(u-u0)
        if k >= maxiter:
            break
        u0 = u #Moves the iteration along
    return u


#test that everything works as it should below: 
A = np.array([
    [5, 2, 1, 1],
    [2, 6, 2, 1],
    [1, 2, 7, 1],
    [1, 1, 2, 8]
])
b = np.array([29, 31, 26, 19])

u = jacobi(A, b, np.zeros_like(b), 1e-10, 1000)
print(u)

#Exact solution
x = np.linalg.solve(A,b)
print(x)

#Works great!

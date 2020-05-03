import numpy as np

#Exactly the same as SOR, but with omega = 1. 


#Forward_subs is used instead of finding inverse of matrix "the old way".
def forward_subs(LU,b):
    ''' Forover substitusjonsalgoritme
    Input:
        LU inneholder både L og U, selv om kun L trengs i denne rutinen
        b Vektor med høyresiden i problemet som skal løses
    Output:
        u Løsningen av det lineære nedretriangulære systemet Lc=b
    '''
    n, m = LU.shape
    u = np.zeros(n)
    u[0] = b[0]/LU[0,0]
    for i in range(1,n):
        u[i] = (b[i]-LU[i,:i] @ u[:i])/LU[i,i]
        
    return u

def Gauss_seidel(A, b, u0, tol, maxiter):
    ''' 
    Gauss-Seidel iterative method. 
    Return the computed solution u.
    u0: The initial value for the iteration
    tol: The tolerance to be used in the stopping criterion (est < tol)
    maxiter: The maximum number of iterations
    '''
    k = 0
    est = 2*tol
    L = np.tril(A,-1)
    D = np.diag(np.diag(A))
    U = np.triu(A,1)
    dividend = L+D
    #dividend_inv = np.linalg.inv(dividend) #This can be used for the solution without using forward_subs (numpy calculates inverse)

    while est > tol:
        print("Iteration ",k,u0)
        k += 1   
        x = -U@u0+b

        #u = dividend_inv@(((1-omega)*D@u0)-(omega*(U@u0))+omega*b)

        #In case using numpy to find inverse is considered "cheating", I solved it wiht forward_substitution instead/also
        u = forward_subs(dividend,x)

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

u = Gauss_seidel(A, b, np.zeros_like(b), 1e-10, 1000)
print(u)

#Exact solution
x = np.linalg.solve(A,b)
print(x)

#Works great!

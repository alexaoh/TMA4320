#Cholesky factorization for positive definite matrix (A=LL^T), and forward + backward substitution to solve Ax = b

import numpy as np
import matplotlib.pyplot as plt

def cholesky(A):
    ''' A er en positiv definitt matrise som returnerer en nedretriangulær matrise L slik at A=L L^T
    '''
    n, m = A.shape
    L = np.zeros((n,m))
    for k in range(n):
        for j in range(k):
            L[k,j] = (A[k, j] - L[k,:j] @ L[j,:j])/L[j,j]
        L[k,k] = (A[k,k] - L[k,:k]@L[k,:k])**(1/2)
    return L

def substit(L,b):
    '''For en nedretriangulær matrise L (nxn) og en vektor b (nx1) finn først c (nx1) slik at Lc=b
    og deretter x (nx1) slik at L^Tx=c'''
    n, m = L.shape
    nb = len(b)
    assert(n == nb) #Verify that the dimensioans are correct
    
    #Find c first
    c = np.zeros(n)
    c[0] = b[0]/L[0,0]
    for i in range(1,n):
        c[i] = (b[i]-L[i,:i] @ c[:i])/L[i,i]
    
    #Find x afterwards
    x = np.zeros(n)
    x[-1] = c[-1]/L[-1,-1]
    for i in range(n-2,-1,-1):
        x[i] = (c[i] - L[i+1:,i] @ x[i+1:])/L[i,i]
    return x


#Example of use: 


A=np.array([[1,2,3],[2,5,4],[3,4,14]])
b = np.array([-2,-8,3])


L = cholesky(A)
x = substit(L, b)

print('L=',L)
print('x=',x)

#Verify that it works! Great!
x = np.linalg.solve(A,b)
print("Solution from np.linalg.solve:",x)

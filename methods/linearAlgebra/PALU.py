import numpy as np
import forwardBackwardSubs as subs

def mylu(A):
    '''Returns: 
    Vector P interpreted as a Pivot matrix (represents a matrix with unit vectors e_Pk in row k).
    LU is a copy of A, where the diagonal and above is U and below the diagonal is L. 
    '''
    LU = A.astype(float) #Copies A and casts all values in A to float! (Important!)
    n = A.shape[0]
    P = np.arange(n)
    for k in range(n-1):
        pivot = np.argmax(abs(LU[P[k:], k]))+k
        P[pivot], P[k] = P[k], P[pivot]
        mults = LU[P[k+1:],k] / LU[P[k],k]
        LU[P[k+1:], k+1:] = LU[P[k+1:], k+1:] - np.outer(mults, LU[P[k],k+1:])
        LU[P[k+1:], k] = mults
    return LU, P

#Example of use: 
A = np.array([[1,1,-2], 
             [1,3,-1], 
             [1,5,1]])
LU, P = mylu(A) 
print(LU)
print(P)
print(LU[P,]) #Prints LU sorted, given by P. 

A2=np.array([[0.3050, 0.5399, 0.9831, 0.4039, 0.1962],
                [0.2563, -0.1986, 0.7903, 0.6807, 0.5544],
                [0.7746, 0.6253, -0.1458, 0.1704,  0.5167],
                [0.4406, 0.9256, 0.4361, -0.2254, 0.7784],
                [0.4568, 0.2108, 0.6006, 0.3677, -0.8922]])
b=np.array([0.9876,-1.231,0.0987,-0.5544,0.7712])

LU2, P2 = mylu(A2)

c = subs.forward_subs(LU2, P2, b)
#print(c)
x = subs.backward_subs(LU2, P2, c)
#print(x)

#Verify that the solution works with this solution from Numpy

import numpy.linalg as lg
x2 = lg.solve(A2,b)
#print(x2)
#Great!

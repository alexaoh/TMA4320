import numpy as np

def mylu(A):
    ''' Returns: 
    Vector P interpreted as a Pivot matrix (represents a matrix with unit vectors e_Pk f'in row k).
    LU is a copy of A, where the diagonal and above is U and below the diagonal is L. 
    '''
    LU = A.astype(float) #Copies A and casts all values in A to float! (Important!)
    n = A.shape[0]
    P = np.arange(n)
    for k in range(n-1):
        #print(LU)
        pivot = np.argmax(abs(A[P[k:], k]))+k
        P[pivot], P[k] = P[k], P[pivot]
        mults = LU[P[k+1:],k] / LU[P[k],k]
        LU[P[k+1:], k+1:] = LU[P[k+1:], k+1:] - np.outer(mults, LU[P[k],k+1:])
        LU[P[k+1:], k] = mults
    return LU, P
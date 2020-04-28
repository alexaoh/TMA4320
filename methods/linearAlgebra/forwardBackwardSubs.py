#Forward substitution (Lc = Pb) and backward substitution (Ux = c) after LU-factorization
import numpy as np
def forward_subs(LU,P,b):
    ''' Forover substitusjonsalgoritme
    Input:
        LU inneholder både L og U, selv om kun L trengs i denne rutinen
        P Permutasjonsvektor av heltall
        b Vektor med høyresiden i problemet som skal løses
    Output:
        c Løsningen av det lineære nedretriangulære systemet Lc=Pb
    '''
    n, m = LU.shape
    Pb = b[P]
    c = np.zeros(n)
    c[0] = Pb[0]
    for k in range(1,n):
        c[k] = Pb[k] - LU[P[k],0:k] @ c[0:k]
        
    return c

def backward_subs(LU,P,c):
    ''' Bakover substitusjonsalgoritme
    Input:
        LU inneholder både L og U, selv om kun U trengs i denne rutinen
        P Permutasjonsvektor av heltall
        c Vektor med høyreside, dvs rutinen løser Ux=c
    Output:
        x Løsningen av det lineære øvretriangulre problemet Ux = c
    '''
    n,m = LU.shape
    x = np.zeros(n)
    x[n-1] = c[n-1]/LU[P[n-1],n-1]
    for k in range(n-1,0,-1):
        x[k-1] = (c[k-1]-LU[P[k-1],k:] @ x[k:])/LU[P[k-1],k-1]
        
    return x
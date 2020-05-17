#Bogacki and Shampine's embedded pair 3(2). 

import numpy as np
import matplotlib.pyplot as plt

def BogackiShampine(t0,tend,y0,f,h0,tol, alpha=0.9):
    T = [t0, 0]
    Y = np.empty((len(y0),2)) 
    stats = {"steps": 0, "feval": 1, "rejects": 0}
    hn = h0
    n = 0
    Y[0] = y0
    k1 = f(t0, y0)
    while tend - T[n] > 0:
        hn = min(hn, tend-T[n])
        k2 = f(T[n]+0.5*hn, Y[n]+0.5*hn*k1)
        k3 = f(T[n]+(3/4)*hn, Y[n]+(3/4)*hn*k2)
        Y[n+1] = Y[n] + (1/9)*hn*(2*k1+3*k2+4*k3)
        T[n+1] = T[n] + hn
        k4 = f(Y[n]+hn,Y[n+1])
        zn1 = Y[n] + (1/24)*hn*(7*k1+6*k2+8*k3+3*k4)
        est = np.linalg.norm(Y[n+1]-zn1)
        if est < tol:
            n += 1
            k1 = k4
            #Could improve: The last point not added (since it is zero and y0). 
            T.append(0)
            Y = np.vstack((Y, y0)) 
            stats["steps"] += 1
        else: 
            stats["rejects"] += 1
        hn*=alpha*(tol/est)**(1/3)
        stats["feval"] += 3
        
    T = np.array(T)
    return T,Y,stats

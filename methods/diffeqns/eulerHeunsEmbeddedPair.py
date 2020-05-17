#Heun's method used to take step, Eulers method used to calculate est in step size control
#(adaptive step size). Therefore a so-called embedded pair of Heun's and Euler's methods 2(1).

import numpy as np
import matplotlib.pyplot as plt

def ImprovedEuler(t0,tend,y0,f,h0,tol, alpha = 0.9):
    h = h0
    k = 0
    F1 = f(t0, y0)
    stats = {"steps": 0, "fevals": 1, "rejects": 0}
    Y = np.zeros((len(y0),2))
    Y[0] = y0
    T = np.array([t0, 0], dtype="float") #Important to specify float!
    while tend - T[k] > 0:
        h = min(h, tend-T[k])
        T[k+1] = T[k]+h
        F2 = f(T[k+1], Y[k]+h*F1)
        Y[k+1] = Y[k] + (h/2)*(F1+F2)
        est = np.linalg.norm((h/2)*(F1-F2))
        if est < tol:
            k += 1
            F1 = f(T[k], Y[k])
            if tend - T[k] > 0: #Bare for å unngå å fjerne siste punkt fra Y og T ved plotting 
                                #(fordi det dermed ikke legges til en siste random verdi i Y og T i siste iterasjon). 
                Y = np.vstack((Y, y0))
                T = np.append(T, 0)
            stats["steps"] += 1
            stats["fevals"] += 1
        else: 
            stats["rejects"] += 1
        stats["fevals"] += 1
        h *= alpha*np.sqrt(tol/est) 
        
    return T,Y,stats

#Specific example with Implicit Euler and Newton's method in each step
#(Have a look at H2019Exam.ipynb for more details if interested)

import numpy as np
import matplotlib.pyplot as plt

def beuler_step(yn,P,h,tol):
    ''' Ta ett skritt med baklengs Euler der 
    yn: initialverdien til skrittet
    P: Et objekt med to medlemsfunksjoner
        P.f: funksjon som tar ett argument y (numpy array) og returnerer dy=y'=f(y)  (numpy array)
        P.Jf: funksjon som tar ett argument y (numpy array) og returnerer jacobi-matrisen J_f(y) (2D numpy array)
    h: Skrittlengde
    tol: toleranse i Newtoniterasjonen
    Returnerer konvergert løsning etter ett skritt
    '''
    #antar at yn er et numpy array
    errest = 2*tol
    z0 = yn
    m = len(yn)
    while errest > tol:
        A = np.eye(m,m) - h*P.Jf(z0)
        b = yn + h*P.f(z0) - z0
        delta = np.linalg.solve(A,b)
        errest = np.linalg.norm(delta)
        z = z0 + delta
        z0 = z
    return z

def beuler(y0,P,h,tol,nsteps):
    ''' Ta nsteps skritt med baklengs Euler
    y0: Startverdien, numpy-array
    P: Objekt som beskrevet i beuler_step
    h: Skrittlengde brukt i integrasjonen (se beuler_step)
    tol: Toleranse brukt i Newtoniterasjonen (se beuler_step)
    nsteps: Antall skritt i integrasjonen, løsningsintevallet er [0,h*nsteps]
    Returner Y, et 2x(nsteps+1) numpy array der kolonne k er [yk, yk']^T, numerisk løsning ved tid t=k*h
    '''
    m = len(y0)
    Y = np.zeros((m, nsteps+1))
    Y[:, 0] = y0
    for step in range(1, nsteps+1):
        Y[:,step] = beuler_step(Y[:,step-1], P, h, tol)    
    return Y

#Used to solve a Van der Pol oscillator, in H2019Exam. 

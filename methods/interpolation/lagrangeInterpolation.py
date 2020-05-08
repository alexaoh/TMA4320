#Lagrange interpolation.

import numpy as np
import matplotlib.pyplot as plt
from newtonInterpolation import newton_polynomial #test Newtons vs Lagrange

def cardinal_functions(xdata, x):
    ''' 
    Inout: 
    xdata: x coordinates of interpolation values.
    x: x value to evaluate the interpolation polynomial in.

    Output: 
    L: list of cardinal functions evaluated in x
    ''' 
    n = len(xdata)
    L = np.zeros((n,len(x)))
    for i in range(n):
        l = 1
        for j in range(n):
            if i is not j:
                l *= (x-xdata[j])/(xdata[i]-xdata[j])
        L[i,:] = l
    return L


def lagrange_polynomial(ydata, xdata, x):
    ''' 
    Inout: 
    ydata: y coordinates of interpolation values.
    xdata: x coordinates of interpolation values.
    x: x value to evaluate the interpolation polynomial in.    

    Output: 
    y: Lagrange polynomial in the given x value. 
    ''' 
    L = cardinal_functions(xdata, x)
    n = len(ydata)
    m = len(xdata)
    assert(n == m)
    y = 0
    for i in range(n):
        y += ydata[i]*L[i,:]
    return y

def plotPointsAndPolynomial(ydata, xdata, x, y, Ny):
    ''' 
    Plots interpolation polynomial with the interpolation points. 
    ''' 
    plt.plot(x,y, 'k',label="Lagrange")
    plt.plot(x,Ny, '--', c="yellow", label="Newton")
    plt.scatter(xdata, ydata, c="red",label="Points")
    plt.legend()
    plt.show()
    

#Test the functions (Example 3.2, page 141 in Sauer):
xdata = np.array([0,1,2,3])
ydata = np.array([2,1,0,-1])
x = np.linspace(-2,len(xdata) + 2,50)

y = lagrange_polynomial(ydata, xdata, x)
Ny = newton_polynomial(xdata, ydata, x)
plotPointsAndPolynomial(ydata, xdata, x, y, Ny)

#Seems to work just fine!
#Newton and Lagrange obviously gives the same interpolation polynomial. Great stuff!

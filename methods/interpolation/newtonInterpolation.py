#Newton's divided differences.

import numpy as np
import matplotlib.pyplot as plt

def N_divided_differences(xdata, ydata):
    ''' 
    Inout: 
    xdata: x coordinates of interpolation values.
    ydata: y coordinates of interpolation values.

    Output: 
    top_row: Top row of f, which is a table of Newton's divided differences (nxn matrix), with x-values in first column.  
    ''' 
    n = len(xdata)
    m = len(ydata)
    assert(n == m)
    f = np.zeros((n, n+1)) #Make table for the differences. 

    for i in range(n): 
        f[i,1] = ydata[i]
    
    for i in range(n): #Add x-values to first column for calculation purposes. 
        f[i, 0] = xdata[i]

    for col in range(2,n+1):
        for row in range(n+1-col):
            f[row,col] = (f[row+1,col-1]-f[row,col-1])/(f[row+col-1, 0]-f[row, 0])

    return f[0,1:] #No need to return the entire matrix, only returning top row. 

def newton_polynomial(xdata, ydata, x):
    ''' 
    Inout: 
    ydata: y coordinates of interpolation values.
    xdata: x coordinates of interpolation values.
    x: x value to evaluate the interpolation polynomial in.  

    Output: 
    y: Newton polynomial in the given x value.  
    ''' 
    top_row = N_divided_differences(xdata, ydata)
    n = len(ydata)
    pol = top_row[0]
    for i in range(1,n):
        factor = 1
        for j in range(i): #Prøv å få til matrisemultiplikasjon @ for å unngå denne løkka!
            factor *= (x-xdata[j])
        pol += top_row[i]*factor

    return pol

#Test, example from notes, compared to the polynonial f found by Newton's divided differences calculated by hand.

def f(x):
    return x**2-x-1

xdata = np.array([0,1,3])
ydata = np.array([-1,-1,5])
x = np.linspace(-2,4,50)

pol= newton_polynomial(xdata, ydata, x)

plt.plot(x, pol, 'k', label="Newton")
plt.plot(x, f(x), '--' ,c="white", label="Notes")
plt.scatter(xdata, ydata, c="red",label="Points")
plt.legend()
plt.show()

#Works just fine!
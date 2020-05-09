#Interpolation nodes which give smallest interpolation error. Combats Runge's fenomena. 
import numpy as np
import matplotlib.pyplot as plt


def chebyshev_nodes(a, b, n):
    ''' n Chebyshev nodes in the interval [a,b]. (Roots of Chebyshev polynomials) ''' 
    i = np.linspace(0,n,n+1)     
    print(i)     
    x = np.cos((2*i+1)*np.pi/(2*(n)))         # nodes over the interval [-1,1]
    return 0.5*(b-a)*x+0.5*(b+a)        # nodes over the interval [a,b]

def chebyshev_polynomial(x, n):
    ''' Gives Chebyshev polynomial #n '''
    return np.cos(n*np.arccos(x))


#Prints first 3 polynomials with there roots, which are Chebyshev nodes!
a = -1
b = 1
x = np.linspace(a,b,100)
plt.plot(x, chebyshev_polynomial(x,0), label="n="+str(0))
plt.plot(x, chebyshev_polynomial(x,1), label="n="+str(1))
plt.plot(chebyshev_nodes(a,b,1), chebyshev_polynomial(chebyshev_nodes(a,b,1), 1), 'o')
plt.plot(x, chebyshev_polynomial(x,2), label="n="+str(2))
plt.plot(chebyshev_nodes(a,b,2), chebyshev_polynomial(chebyshev_nodes(a,b,2), 2), 'o')
plt.legend()
plt.show()

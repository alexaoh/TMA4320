#Composite trapezoid quadrature implementation

def trapezoid(f,a,b,n):
    h = float(b-a)/n
    result = (f(a) + f(b))/2.0
    for k in range(1,n):
        result += f(a + k*h)
    return h*result

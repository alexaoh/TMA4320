#Bisection method (finding root of continuous function)

def bisection(f,a,b,tol):
    ant_halv = 0  
    while tol < abs(b-a): 
        #using this condition since tol < (b-a) => m < (1/2)(b-a). 
        m = (1/2)*(a+b)
        fm = f(m)
        if fm == 0:
            return m, ant_halv
        elif f(a)*fm<0:
            b = m
        else:
            a = m
        ant_halv += 1
    return m, ant_halv 

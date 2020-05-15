# Runge-Kutta of fourth order (Kutta 1901) with fixed step length and amount of steps

def kutta4_step(f, t, y, h):
    f1 = f(t, y)
    f2 = f(t+(h/2), y+(h/2)*f1)
    f3 = f(t+(h/2), y+(h/2)*f2)
    f4 = f(t+h, y+h*f3)
    return y + (h/6)*(f1+2*f2+2*f3+f4)
    

def kutta4(f,t0,y0,h,nsteps):
    m = len(y0)
    Y = np.zeros((nsteps+1,m))
    T = np.zeros(nsteps+1)
    T[0] = t0
    Y[0] = y0
    for i in range(nsteps):
        Y[i+1] = kutta4_step(f, T[i], Y[i], h)
        T[i+1] = T[i] + h
    return T,Y

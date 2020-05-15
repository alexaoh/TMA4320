#Euler's method with fixed step length and amount of steps

def heuns_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t+h, y + k1*h)
    return y + (1/2)*(k1+k2)

def heuns_method(f, t0, y0, h, nsteps):
    m = len(y0)
    Y = np.zeros((nsteps+1,m))
    T = np.zeros(nsteps+1)
    T[0] = t0
    Y[0] = y0
    for i in range(nsteps):
        Y[i+1] = heuns_step(f, T[i], Y[i], h)
        T[i+1] = T[i] + h
    return T, Y

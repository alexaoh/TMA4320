import numpy as np
import matplotlib.pyplot as plt

def step_euler(y,t, h, f, n):
    ''' Performs one step of Euler's method. '''
    return y + h*f(y,t, n)

def step_RK4(y,t, h, f, n):
    ''' Calculates all k's and takes on RK4-step '''
    k1 = f(y,t, n)
    k2 = f(y + h*k1/2, t+h/2, n)
    k3 = f(y+h*k2/2, t+h/2, n)
    k4 = f(y+h*k3, t+h, n)
    return y + (h/6)*(k1+2*k2+2*k3+k4)

def full_numerical_method(h, f, n, method="euler", y_0 = 1, start_t = 0.01, end_t = 3):
    ''' Performs all steps in selected method. Made one general function for both Euler and RK4 '''
    try:
        t_list = np.zeros(1)
        y_list = np.zeros((1,2))
        t_list[0] = start_t
        y_list[0] = y_0
        i = 0
        next_value = step_RK4(y_list[i], t_list[i], h, f, n)
        if method == "euler":
            while next_value[0] > 0:
                t_list = np.append(t_list, t_list[i]+h)
                y_list = np.vstack((y_list, next_value))
                y_list[i+1] = next_value
                i+=1
                next_value = step_euler(y_list[i], t_list[i], h, f, n)
        elif method == "RK4":
            while next_value[0] >= 0:
                t_list = np.append(t_list, t_list[i]+h)
                y_list = np.vstack((y_list, next_value))
                y_list[i+1] = next_value
                i+=1
                next_value = step_RK4(y_list[i], t_list[i], h, f, n)
                print(t_list[i]+h)
        return t_list, y_list
    except:
        return t_list, y_list

def lane_emden_analytical(t):
    return np.sin(t)/t

def lane_emden(y, t, n):
    assert(y[0]>0) #this exception is caught so that no runtimeerrors are casted!
    return np.array([y[1], -y[0]**(n)-2*y[1]/t])

def simple_plot(t, y, h, n, anal=False, savefig=False, filename="simple_plot.pdf"):
    ''' Simple plot for these tasks. 
    anal = False (default) does not plot the analytical function, True does. 
    savefig = False (default) does not save plot to PDF. True saves to default filename ""simple_plot.pdf"
    n = given as the value you want to use, not the list n.
    '''
    if anal == True:
        plt.plot(t[0], lane_emden_analytical(t[0]), label="Analytisk")
    for i in range(len(h)):
        plt.plot(t[i], y[i][:,0], label="h = "+str(h[i]))
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$\theta(\xi)$')
    plt.title(r'$n = '+str(n)+'$')
    plt.legend()
    if savefig == True:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def print_values(list_of_t_lists, list_of_y_lists, h, n):
    ''' n is supplied not as a list, but as the n that should be displayed'''
    for i in range(len(h)):
        print('ξ_N =',list_of_t_lists[i][-1], "h = "+str(h[i])+", n = "+str(n))
        print("ξ_N²|θ'(ξ_N)|=", list_of_t_lists[i][-1]**2*np.abs((list_of_y_lists[i][:,0][-1]-list_of_y_lists[i][:,0][-2])/h[i]), "h = "+str(h[i])+", n = "+str(n))
        print()


h = [0.01, 0.001, 0.0001]
n = [1, 3/2, 3]

t_listh0_kutta, y_listh0_kutta = full_numerical_method(h[0], lane_emden, n[1], "RK4", 1, 0.01)
t_listh1_kutta, y_listh1_kutta = full_numerical_method(h[1], lane_emden, n[1], "RK4", 1, 0.01)
t_listh2_kutta, y_listh2_kutta = full_numerical_method(h[2], lane_emden, n[1], "RK4", 1, 0.01)
list_of_t_lists_kutta_non = [t_listh0_kutta, t_listh1_kutta, t_listh2_kutta]
list_of_y_lists_kutta_non = [y_listh0_kutta, y_listh1_kutta, y_listh2_kutta]
simple_plot(list_of_t_lists_kutta_non, list_of_y_lists_kutta_non, h, n[1])
print_values(list_of_t_lists_kutta_non, list_of_y_lists_kutta_non, h, n[1])

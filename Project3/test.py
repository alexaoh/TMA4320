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

def lane_emden_analytical(t):
    return np.sin(t)/t

def lane_emden(y, t, n):
    #assert(y[0]>0) #this exception is caught so that no runtimeerrors are casted! (when n=3/2 runtimeerror, since it does not work on negative numbers)
    return np.array([y[1], -y[0]**(n)-2*y[1]/t])

def full_numerical_method(h, n, method="euler", y_0 = [1,0], start_t = 0.01, end_t = 3, f = lane_emden):
    ''' Performs all steps in selected method. Made one general function for both Euler and RK4 '''
    #try:
    t_list = np.zeros(1)
    y_list = np.zeros((1,2))
    t_list[0] = start_t
    y_list[0] = y_0
    i = 0
    next_value = step_RK4(y_list[i], t_list[i], h, f, n)
    if method == "euler":
        while next_value[0] > -1e-04: #Må gå "langt" inn på negativ theta for å få den siste ksi som de skriver i oppgaven!
            t_list = np.append(t_list, t_list[i]+h)
            y_list = np.vstack((y_list, next_value))
            y_list[i+1] = next_value
            i+=1
            next_value = step_euler(y_list[i], t_list[i], h, f, n)
    elif method == "RK4":
        while next_value[0] > 0:
            t_list = np.append(t_list, t_list[i]+h)
            y_list = np.vstack((y_list, next_value))
            y_list[i+1] = next_value
            i+=1
            next_value = step_RK4(y_list[i], t_list[i], h, f, n)
    return t_list, y_list
    #except AssertionError:
    #    print("De negative theta-verdiene tas ikke med")
    #    return t_list, y_list

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
    plt.legend(loc='upper right')
    if savefig == True:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
def get_values(h, n, method="euler", y_0 = [1,0], start_t = 0.01, end_t = 3, f = lane_emden):
    ''' made to get rid of some repetitiveness in gettin values from full_numerical_method for every value of h '''
    list_of_t_lists = []
    list_of_y_lists = []
    for i in range(len(h)):
        t_list, y_list = full_numerical_method(h[i], n, method, y_0, start_t, end_t, f)
        list_of_t_lists.append(t_list)
        list_of_y_lists.append(y_list)
    return list_of_t_lists, list_of_y_lists

def print_values(list_of_t_lists, list_of_y_lists, h, n):
    ''' n is supplied not as a list, but as the n that should be displayed'''
    for i in range(len(h)):
        print('ξ_N =',list_of_t_lists[i][-1], "h = "+str(h[i])+", n = "+str(n))
        print("ξ_N²|θ'(ξ_N)|=", list_of_t_lists[i][-1]**2*np.abs((list_of_y_lists[i][:,0][-1]-list_of_y_lists[i][:,0][-2])/h[i]), "h = "+str(h[i])+", n = "+str(n))
        print()

h = [0.01, 0.001, 0.0001, 0.00001] 
n = [1, 3/2, 3]

#Testing global error code: 

ksiN = [3.65375, 6.89685] #Korrekt ifølge oppgaven, selv om man må "langt" inn på negative theta-verdier for å nå så langt i ksi. 

def fixed_full_euler_or_RK4(h, f, n=n[0], method="euler", y_0 = [1,0], start_t = 0.01, end_t = ksiN[0]):
    #Performs all steps in RK4 or Euler, this time all the way to end_t.
    #try: 
    N = int((end_t - start_t)/h) 
    t_list = np.linspace(start_t, end_t, N+1) 
    y_list = np.zeros((N+1,2))
    t_list[0] = start_t
    y_list[0] = y_0
    if method == "euler":
        for i in range(N):
            next_value = step_euler(y_list[i], t_list[i], h, f, n)
            y_list[i+1] = next_value
            if next_value[0] <= 0:
                print(t_list[i])
    elif method == "RK4":
        for i in range(N):
            next_value = step_RK4(y_list[i], t_list[i], h, f, n)
            y_list[i+1] = next_value
    return t_list, y_list
    #except:
    #    print("AssertionError: De negative theta-verdiene tas ikke med!")
    #    return t_list, y_list

def global_error(theta_list):
    #Finds global error of one theta, which depends on the h-value used to solve the equation 
    return np.abs(theta_list[-1])
    #return np.full(len(h_values),np.abs(y_list[:,0][-1]))

def global_error_vs_h_values(h_values, f, n=n[0], method="euler", y_0 = [1,0], start_t = 0.01, end_t = ksiN[0]):
    
    #Takes y-list from method and h-values. 
    #Returns numpy array of global_error that later may be plotted against h (in correct dimensions)
    
    global_error_list = np.zeros(len(h_values))
    for i in range(len(h_values)):
        t_list, y_list = fixed_full_euler_or_RK4(h_values[i], f, n, method, y_0, start_t, end_t)
        theta_list = y_list[:,0]
        global_error_list[i] = global_error(theta_list)
    return global_error_list
        

def plot_y_vs_h(y_list, h_values, title, label=r'$|\theta_N|$', xlabel=r'h', ylabel=r'$|\theta_N|$'):
    plt.plot(h_values, y_list, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log')
    plt.legend()
    plt.show()

h_values = np.linspace(1e-5,0.1, 100) #E.g. 
h_values_log = np.logspace(-4,0,100) #Try with log-scale too.
n = [3/2, 3]


global_errorsE1 = global_error_vs_h_values(h_values, lane_emden, n[0], "euler", [1,0], 0.01, ksiN[0])
#global_errorsE2 = global_error_vs_h_values(h_values_log, lane_emden, n[1], "euler", [1,0], 0.01, 6.8)
plot_y_vs_h(global_errorsE1, h_values, r'Euler, $n='+str(n[0])+'$')
#plot_y_vs_h(global_errorsE2, h_values, r'Euler, $n='+str(n[1])+'$')

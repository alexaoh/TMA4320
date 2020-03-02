from loader import *
from algorithm import *



I = 12223 #images --> kjører et beskjedent antall blant de 60000, basert på Stochastic Gradient Descent. 
#Blir ca 12000 bilder maksimalt med kun utvalg på to siffer. 
d = 784 # 28x28
Y_0, c = get_dataset()
#Y_Kk, J, omega, my, iterations = algorithm(Y_0, c, I, d)

Y0_chunk, chunk = stochastic_gradient_descent(I, Y_0)

from numpy import *
from math import *
from matplotlib.pylab import *

t=linspace(0,1000,101)
m=1
tau=1
V_0=0.00001
a_dot= -V_0*sin(30*t)*exp(t/100)*899.99 + (3/5)*V_0*cos(30*t)*exp(t/100)
F=m*tau*a_dot
xlabel('time---->')
ylabel(' radiation reaction force---->')
title('Radiation reaction force as a function of time')
plot(t,F)
show()

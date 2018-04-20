#Script for CVaR Value Iteration for small grid world domain
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from Gridworld_MDP_Pen_class import Gridworld_MDP_Pen_class
from VI_Exp import VI_Exp
from VI_CVaR import VI_CVaR

#Grid-world parameters
eps = 0.05;         #noise parameter
target_row = 1;  #0	#target location
target_col = 15; #4
mdp = Gridworld_MDP_Pen_class('instances/gridworld_16x14.png',eps,target_row,target_col,'file',-2);
# ----------------------------------  Do standard value iteration ---------------------
maxIter = 1e3;
tol = 1e-5;
dis = 0.95;

[V_Exp,Pol_Exp,err_Exp] = VI_Exp(mdp,tol,maxIter,dis);
print 'Standard value iteration complete.'
# show values on image, and sample trajectory
start_row = 12; #4
start_col = 15;	#4
im = mdp.val2image(-V_Exp[0:np.size(V_Exp)-1]);
plt.figure;plt.imshow(im, cmap=plt.cm.get_cmap('Blues', 6));

plt.colorbar;
plt.clim(-2, 1);
[row,col] = mdp.getPath(Pol_Exp[0:np.size(Pol_Exp)-1],start_row,start_col,int(maxIter));
#plt.plot(col,row,'k-x', linewidth=5.0,markersize=15.0);
#plt.show()


# -----------------------------------  Do CVaR Value Iteration -------------------------
#Choose solver: currently only CPlex is supported.
#linprog' - CPlex
#addpath('/Applications/CPLEX_Studio128/cplex/matlab/x86-64_osx')
index_opt = 'linprog';
#set VI parameters
Ny = 21;        # number of interpolation points for y
log_func = np.logspace(-2,0,Ny-1)
log_func = np.insert(log_func,0,0)
Y_set_all = np.ones([mdp.getNumStates(),1])*log_func;
maxIter = 40;
tol = 1e-3;
#Do CVaR VI
print('Performing CVaR value iteration...');
[V_CVaR,Pol_CVaR,err_CVaR] = VI_CVaR(mdp,Y_set_all,index_opt,0,tol,maxIter,matlib.repmat(V_Exp,1,Ny),dis);
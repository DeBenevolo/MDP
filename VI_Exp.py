import numpy as np
from copy import copy, deepcopy

def VI_Exp(MDP,tol,maxIter,dis):
	#Initialize
	Ns = MDP.getNumStates();
	maxIter = int(maxIter)
	V=np.zeros((Ns,1));
	#print np.size(MDP.P, axis=0)
	Pol = np.zeros((Ns,1));
	# Do Value Iteration
	for i in range(maxIter):
		V_prev = deepcopy(V)
    		for s in range(Ns):
        		a = MDP.getActions(s);
        		Q = -MDP.getReward(s,a) + dis*np.dot(MDP.nextStateProb(s,a), V_prev);
        		V[s] = np.min((Q)); #minimum element of each column
        		Pol[s] = np.argmin(Q); #minimum element of each column  

    		err = np.max(np.absolute(V-V_prev))
    		print err;
    		if err < tol:
    			print "Error threshold reached at iteration: ", err, i;
    			return V, Pol, err

	return V, Pol, err
import abc
import numpy as np
from MDP_class import MDP_class

class Finite_MDP_class(MDP_class):

    def __init__(self,P,R,A):
        #P is Ns*Ns*Na matrix of transitions P(s'|s,a)
        #R is Ns*Na matrix of deterministic rewards r(s,a)
        #A is Ns*Na binary matrix of available actions at each state
        if type(P) is np.ndarray:
            self._P = P
        else:
            self._P = np.array(P)

        if type(R) is np.ndarray:
            self._R = R
        else:
            self._R = np.array(R)

        if type(A) is np.ndarray:
            self._A = A
        else:
            self._A = np.array(A)

        self._Ns = np.size(self._P, axis=0)
        self._Na = np.size(self._P, axis=2)


    @property
    def P(self):
        return self._P

    @property
    def R(self):
        return self._R

    @property
    def A(self):
        return self._A

    @property
    def Ns(self):
        return self._Ns

    @property
    def Na(self):
        return self._Na
    
    def getNumStates(self):
        return np.size(self._P, axis=0)
    
    def getNumActions(self):
        return self._Na
    
    def getActions(self,s):
        return self._A[s,].ravel(order='F').nonzero()
    
    def getReward(self,s,a):
        # a is a array 
        # s is number
        return self._R[s,a].T
    
    def nextStateProb(self,s,a):
        # get next state probability for action a
        # if a is a scalar the function returns a row vector
        # if a is a vector then a matrix is returned with the
        # probabilities on rows
        if np.size(a) == 1:
            return np.squeeze(self._P[s,:,a])
        else:
            return np.squeeze(self._P[s,:,a])

    
    def sampleNextState(self,s,a):
        return None


#MDP_class.register(Finite_MDP_class)

#f __name__ == '__main__':
 #   print 'Subclass:', issubclass(Finite_MDP_class, MDP_class)
  #  P = np.arange(2500).reshape(25, 25, 4)
  #  R = np.arange(100).reshape(25, 4)
   # A = np.arange(100).reshape(25, 4)
   # mdp = Finite_MDP_class(P, R, A)
   # print 'Instance:', isinstance(mdp, MDP_class)
    #a = mdp.getActions(1)
   # print 'getActions:', a
   # print 'getReward:', mdp.getReward(1, a)
   # next1 = mdp.nextStateProb(1, a)
   # print 'nextStateProb:', next1
   # print 'nextStateProb:', np.size(next1, axis=1)

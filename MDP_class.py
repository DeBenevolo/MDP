import abc
class MDP_class(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getNumStates(self):
        return
    
    @abc.abstractmethod
    def getNumActions(self):
        return
    
    @abc.abstractmethod
    def getActions(self,s):
        return
    
    @abc.abstractmethod
    def getReward(self,s,a):
        return
    
    @abc.abstractmethod
    def nextStateProb(self,s,a):
        return
    






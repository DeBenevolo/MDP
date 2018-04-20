import abc
import numpy as np
from PIL import Image
from Finite_MDP_class import Finite_MDP_class

class Gridworld_MDP_Pen_class(Finite_MDP_class):

    def __init__(self,ImageFile,eps,targetRow,targetCol,image_type,penalty):
        self._targetRow = targetRow
        self._targetCol = targetCol
        if image_type == 'file':
            self._img = np.asarray(Image.open(ImageFile).convert('L'))
        else:
            raise ValueError('unknown image type')

        self._Nrow = np.size(self._img,0);
        self._Ncol = np.size(self._img,1);
        self._obstacles = (self._img == 0).ravel(order='F').nonzero()
        self._nonobstacles = (self._img != 0).ravel(order='F').nonzero()


        self._Ns = self._Nrow*self._Ncol;
        self._Na = 4;
        self._NumState = self._Ns+1;

        # last state is the die-state
        self._die_state = self._NumState-1;

        self._Pn = np.zeros((self._NumState, self._NumState)); # north
        self._Ps = np.zeros((self._NumState, self._NumState)); # south
        self._Pe = np.zeros((self._NumState, self._NumState)); # east
        self._Pw = np.zeros((self._NumState, self._NumState)); # west

        self._R = -1*np.ones((self._NumState,self._Na));

        self._gridworld_matrix_shape = [self._Nrow, self._Ncol]   
        for i in range(self._Nrow):
            for j in range(self._Ncol):
                #current position in state form
                curpos = self.sub2ind(self._gridworld_matrix_shape, i, j);  #linear index
                #generate next positions(north, south, east and west)
                [newrow,newcol,newcollision_index] = Gridworld_MDP_Pen_class.neigborhood_fn(i,j,self.Nrow,self.Ncol,self._img);
                nextpos = np.zeros((4))
                for kk in range(len(newrow)):
                    if np.isfinite(newrow[kk]) == True and np.isfinite(newcol[kk]) == True:
                        #does not hit obstacles
                        nextpos[kk] = self.sub2ind(self._gridworld_matrix_shape, int(newrow[kk]), int(newcol[kk]));
                    else:
                        #hit obstacles
                        nextpos[kk] = self._die_state;


                nextpos_north = int(nextpos[0]);
                nextpos_south = int(nextpos[1]);
                nextpos_east = int(nextpos[2]);
                nextpos_west = int(nextpos[3]);

               
                self._Pn[curpos,nextpos_north] = self._Pn[curpos,nextpos_north] + 1-eps;
                self._Pn[curpos,nextpos_south] = self._Pn[curpos,nextpos_south] + eps/3;
                self._Pn[curpos,nextpos_east] = self._Pn[curpos,nextpos_east] + eps/3;
                self._Pn[curpos,nextpos_west] = self._Pn[curpos,nextpos_west] + eps/3;
                    
                self._Ps[curpos,nextpos_north] = self._Ps[curpos,nextpos_north] + eps/3;
                self._Ps[curpos,nextpos_south] = self._Ps[curpos,nextpos_south] + 1-eps;
                self._Ps[curpos,nextpos_east] = self._Ps[curpos,nextpos_east] + eps/3;
                self._Ps[curpos,nextpos_west] = self._Ps[curpos,nextpos_west] + eps/3;
                    
                self._Pe[curpos,nextpos_north] = self._Pe[curpos,nextpos_north] + eps/3;
                self._Pe[curpos,nextpos_south] = self._Pe[curpos,nextpos_south] + eps/3;
                self._Pe[curpos,nextpos_east] = self._Pe[curpos,nextpos_east] + 1-eps;
                self._Pe[curpos,nextpos_west] = self._Pe[curpos,nextpos_west] + eps/3;
                    
                self._Pw[curpos,nextpos_north] = self._Pw[curpos,nextpos_north] + eps/3;
                self._Pw[curpos,nextpos_south] = self._Pw[curpos,nextpos_south] + eps/3;
                self._Pw[curpos,nextpos_east] = self._Pw[curpos,nextpos_east] + eps/3;
                self._Pw[curpos,nextpos_west] = self._Pw[curpos,nextpos_west] + 1-eps;
        
        #recurrent state at target
        target = self.sub2ind(self._gridworld_matrix_shape,self._targetRow, self._targetCol);
        self._Pn[target,:] = 0;
        self._Pn[target,target] = 1;
        self._Ps[target,:] = 0;
        self._Ps[target,target] = 1;
        self._Pe[target,:] = 0;
        self._Pe[target,target] = 1;
        self._Pw[target,:] = 0;
        self._Pw[target,target] = 1;
            
        self._R[target,:] = 0;
            
        #die_state (state NumState) is recurrent as well recurrent state at die state
        self._Pn[self._die_state,:] = 0;
        self._Pn[self._die_state, self._die_state] = 1;
        self._Ps[self._die_state,:] = 0;
        self._Ps[self._die_state,self._die_state] = 1;
        self._Pe[self._die_state,:] = 0;
        self._Pe[self._die_state,self._die_state] = 1;
        self._Pw[self._die_state,:] = 0;
        self._Pw[self._die_state,self._die_state] = 1;
            
        self._R[self._die_state,:] = penalty;
            
        #add the die_state back to the system
        nonobstacles_total = np.concatenate((self._nonobstacles[0], [self._die_state])); 
        #states and rewards only define for nonobstacles
        self._Pn = self._Pn[nonobstacles_total,:]
        self._Pn = self._Pn[:,nonobstacles_total]
        self._Ps = self._Ps[nonobstacles_total,:];
        self._Ps = self._Ps[:,nonobstacles_total];
        self._Pe = self._Pe[nonobstacles_total,:];
        self._Pe = self._Pe[:,nonobstacles_total];
        self._Pw = self._Pw[nonobstacles_total,:];
        self._Pw = self._Pw[:,nonobstacles_total];
 
            
        #with four actions: north south east west
        self._P = np.zeros((len(nonobstacles_total), len(nonobstacles_total), 4));

        self._P[:,:,0] = self._Pn;
        self._P[:,:,1] = self._Ps;
        self._P[:,:,2] = self._Pe;
        self._P[:,:,3] = self._Pw;
            
        # reward for nonobstacles
        self._R = self._R[nonobstacles_total,:];
        self._A = np.ones((len(self._P[0]),self._Na));
        Finite_MDP_class(self._P, self._R, self._A)
                      


    def sub2ind(self, array_shape, rows, cols):
        ind = cols*array_shape[0] + rows
        return ind

    def ind2sub(self, array_shape, ind):
        cols = (int(ind) / array_shape[0])
        rows = (int(ind) % array_shape[0])
        return rows, cols

    @property
    def Nrow(self):
        return  self._Nrow

    @property
    def Ncol(self):
        return  self._Ncol

    @property
    def img(self):
        return self._img

    @property
    def obstacles(self):
        return self._obstacles

    @property
    def nonobstacles(self):
        return self._nonobstacles

    @property
    def targetRow(self):
        return self._targetRow

    @property
    def targetCol(self):
        return self._targetCol

    @property
    def die_state(self):
        return self._die_state

    @property
    def eps(self):
        return self._eps

    @staticmethod
    def north(row,col,Nrow,Ncol,im):
    	collision_index = 0;
        newrow = np.maximum(row-1,0);
        newcol = col;
            
        # if we hit an obstacle   collision_index = 1;
        if im[newrow, newcol] == 0:
            collision_index = 1;
            newrow = np.inf;
            newcol = np.inf;

        return newrow, newcol, collision_index

    @staticmethod
    def south(row,col,Nrow,Ncol,im):
        collision_index = 0;
        newrow = min(row+1,Nrow-1);
        newcol = col;
            
        # if we hit an obstacle   collision_index = 1;
        if im[newrow, newcol] == 0:
            collision_index = 1;
            newrow = np.inf;
            newcol = np.inf;

        return newrow, newcol, collision_index

    @staticmethod
    def east(row,col,Nrow,Ncol,im):
        collision_index = 0; 
        newrow = row;
        newcol = min(col+1,Ncol-1);
            
        # if we hit an obstacle   collision_index = 1;
        if im[newrow, newcol] == 0:
            collision_index = 1;
            newrow = np.inf;
            newcol = np.inf;

        return newrow, newcol, collision_index

    @staticmethod
    def west(row,col,Nrow,Ncol,im):
        collision_index = 0; 
        newrow = row;
        newcol = max(col-1,0);
            
        # if we hit an obstacle   collision_index = 1;
        if im[newrow, newcol] == 0:
            collision_index = 1;
            newrow = np.inf;
            newcol = np.inf;

        return newrow, newcol, collision_index

    @staticmethod
    def neigborhood_fn(row,col,Nrow,Ncol,im):
        #Generate neigbors of state (row,col)
        newrow = np.zeros((4));
        newcol = np.zeros((4));
        newcollision_index = np.zeros((4));

        #north
        [row_buf, col_buf, collision_index_buf]= Gridworld_MDP_Pen_class.north(row,col,Nrow,Ncol,im);
        newrow[0] = row_buf;
        newcol[0]= col_buf;
        newcollision_index[0] = collision_index_buf;
        #south
        [row_buf, col_buf, collision_index_buf]= Gridworld_MDP_Pen_class.south(row,col,Nrow,Ncol,im);
        newrow[1] = row_buf;
        newcol[1] = col_buf;
        newcollision_index[1] = collision_index_buf;
        #east
        [row_buf, col_buf, collision_index_buf]= Gridworld_MDP_Pen_class.east(row,col,Nrow,Ncol,im);
        newrow[2] = row_buf;
        newcol[2] = col_buf;
        newcollision_index[2] = collision_index_buf;
        #west
        [row_buf, col_buf, collision_index_buf]= Gridworld_MDP_Pen_class.west(row,col,Nrow,Ncol,im);
        newrow[3] = row_buf;
        newcol[3] = col_buf;
        newcollision_index[3] = collision_index_buf;

        return newrow,newcol,newcollision_index



    #Graphic 
    def val2image(self,val):
        im = np.zeros((self.Nrow*self.Ncol));
        #define the obstacle to be low reward
        #get rid of the fake recurrent state: die_state
        im[self.nonobstacles[0]] = val[0];
        return im.reshape((self.Nrow, self.Ncol), order='F')

    def getState(self,row,col):
        #return state index for row and column
        if row==0 or col==0:
            s = np.size(self.nonobstacles)+1; #die state
        else:
            aux_nonobstacles = np.array((self.nonobstacles))
            s = (aux_nonobstacles == self.sub2ind([self._Nrow,self._Ncol],row,col)).ravel(order='F').nonzero()
            if np.size(s) == 0:
                s = size(self.nonobstacles)+1; #obstacle - go to die state
        return s

    def getRowCol(self,s):
        #return row and column for state index
        row = np.zeros(np.size(s)); 
        col = row;
        for i in range(np.size(s)):
            if s[i] > np.size(obj.nonobstacles): #die state
                row[i] = 0;
                col[i] = 0;
            else:
                [row[i],col[i]] = self.ind2sub([self._Nrow,self._Ncol],obj.nonobstacles(s[i]));
            
    
    def getPath(self,pol,row0,col0,maxIter):
        #get path until target starting from (row0,col0)
        pol_im_aux = np.zeros((self.Nrow*self.Ncol));
        pol_im_aux[self.nonobstacles] = pol.T;
        
        pol_im = pol_im_aux.reshape((self.Nrow, self.Ncol), order='F')
        row = np.zeros((maxIter,1), dtype=np.int);
        col = np.zeros((maxIter,1), dtype=np.int);
        row[0] = row0;
        col[0] = col0;
        _i =0
        for i in range(1, maxIter):
            _i = i
            index_e =0
            if pol_im[row[i-1],col[i-1]] == 0.0:   #north
                    [row[i],col[i], index_e] = Gridworld_MDP_Pen_class.north(row[i-1],col[i-1],self._Nrow,self._Ncol,self._img);
            elif pol_im[row[i-1],col[i-1]] == 1.0:  #south
                    [row[i],col[i], index_e] = Gridworld_MDP_Pen_class.south(row[i-1],col[i-1],self._Nrow,self._Ncol,self._img);
            elif pol_im[row[i-1],col[i-1]] == 2.0:   #east
                    [row[i],col[i],index_e] = Gridworld_MDP_Pen_class.east(row[i-1],col[i-1],self._Nrow,self._Ncol,self._img);
            elif pol_im[row[i-1],col[i-1]] == 3.0: #west
                    [row[i],col[i], index_e] = Gridworld_MDP_Pen_class.west(row[i-1],col[i-1],self._Nrow,self._Ncol,self._img);
            else:
                print 'unknown action';
            
            if np.isfinite(row[i]) != True  or np.isfinite(col[i]) != True:
                print 'Hit an obstacle, mission failed'
                break;
            if self.targetRow == row[i] and self.targetCol == col[i]:
                print 'Arrive at target, mission accomplised'
                break;
        
        row = row[0:_i+1];
        col = col[0:_i+1];

        return row, col


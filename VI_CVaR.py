import numpy as np
import cplex
from copy import copy, deepcopy
from scipy.sparse import csr_matrix
import scipy.linalg

def VI_CVaR(MDP,Y_set_all,index_fast,tol,maxIter,V0,dis):
    #Interpolation based CVaR VI
    #Instead of solving many small problem, solve a concatenated linear problem
    #for all actions and confidence parameters at each iteration
    Ns = MDP.getNumStates();
    Ny = np.size(Y_set_all);
    Na = MDP.getNumActions();

    V = V0;
    Pol = np.zeros((Ns,Ny));
    warm_start = np.zeros((Ns,Ny,Na), dtype=np.object);
    f_cell = np.zeros((Na,Ny-1), dtype=np.object);
    A_cell = np.zeros((Na,Ny-1), dtype=np.object);
    B_cell = np.zeros((Na,Ny-1), dtype=np.object);
    Aeq_cell = np.zeros((Na,Ny-1), dtype=np.object);
    Beq_cell = np.zeros((Na,Ny-1), dtype=np.object);
    lb_cell = np.zeros((Na,Ny-1), dtype=np.object);
    ub_cell = np.zeros((Na,Ny-1), dtype=np.object);
    xi_t0_cell = np.zeros((Na,Ny-1), dtype=np.object);

    #Set up optimization options and functions
    #options = cplexoptimset;
    #options.Display = 'off';

    #VI algorithm with Policy and Value function outputs
    for i in range(maxIter):
        V_prev = deepcopy(V)
        for s in range(Ns):
            #I get rid of this line for better speed performance
            #disp(['iteration at state: ' num2str(Ns-s)]);
            y_set = Y_set_all #the discretization of y for each state
            a = MDP.getActions(s)
            objective_fn = np.zeros((np.size(a), np.size(y_set)))
            for a_ind in range(np.size(a)):
                trans_prob = MDP.nextStateProb(s,a[0][a_ind]) # return an array in this case
                #intialize nz_prob_ind so it has a fixed length
                nz_prob_ind = -1*np.ones((1, np.size(trans_prob)));
                #optimize only over non-zero probabilities
                ind_prob_pos = (trans_prob > 0.0).ravel(order='F').nonzero()
                nz_prob_ind[0][0:np.size(ind_prob_pos)] = np.array((ind_prob_pos)) #rest is zero
                #number of non-zero elements 
                Ns_nz = np.size(ind_prob_pos)
                lb = np.zeros((Ns_nz,1))
                ub = lambda y_in: np.ones((Ns_nz,1))/y_in
                Aeq = trans_prob[ind_prob_pos]
                Beq = 1

                state_ind = nz_prob_ind
                state_ind = (state_ind != -1.).ravel(order='F').nonzero()

                Aeq = np.reshape(Aeq, (1,np.size(Aeq)))

                Aeq_LP = np.concatenate((Aeq, np.zeros((np.size(Aeq, axis=0), Ns_nz))))
                Beq_LP = Beq

                lb_LP = [[lb],[-1e6*np.ones((Ns_nz, 1))]];
                ub_LP = lambda y_in: [[ub(y_in)], [1e6*np.ones((Ns_nz, 1))]];

                A_LP = np.zeros(((np.size(y_set)-1)*Ns_nz, 2*Ns_nz));
                B_fn = np.zeros((np.size(y_set)-1,1), dtype=np.object)

                for n in range(np.size(y_set)-1):
                    #DELTA Y / DELTA X = SLOPE
                    slope = (y_set[n+1]*V_prev[state_ind,n+1].T-y_set[n]*V_prev[state_ind,n].T)/(y_set[n+1]-y_set[n]); #find the first component of equation 9.5
                    if n == 0:
                        n_b = 0; n_e = Ns_nz;
                    else:
                        n_b = n*Ns_nz; n_e = (n+1)*Ns_nz

                    _trans_prob_aux = trans_prob[ind_prob_pos]                            
                    _slope_aux = (slope*_trans_prob_aux.reshape(np.size(_trans_prob_aux),1))
                    _diag_aux = -np.diag(_slope_aux.reshape((np.size(_slope_aux),)))
                    A_LP[n_b:n_e,:] = np.concatenate((_diag_aux, np.eye(Ns_nz)), axis=1)
                    B_fn[n] = lambda y_in: ((V_prev[state_ind,n].T*y_set[n])/y_in) - ((slope*y_set[n])/y_in)
                   

                f_LP = np.concatenate((np.zeros((1, Ns_nz)), np.ones((1,Ns_nz))), axis=1);

                for y_ind in range(np.size(y_set)):

                    if y_set[y_ind] == 0:
                        # if y=0, we do not need to solve the inner optimization
                        # problem because the corresponding result is V(x,0)
                        objective_fn[a[0][a_ind],y_ind] = np.max(V_prev[ind_prob_pos,y_ind]);

                    else:
                        if np.size(warm_start[s,y_ind,a_ind]) <= 1:
                            warm_start[s,y_ind,a_ind] = np.ones((Ns_nz, 1));
                        
                        xi_0 = warm_start[s,y_ind,a_ind];

                        B_LP = np.zeros(((np.size(y_set)-1)*Ns_nz, 1));
                        for n in range(np.size(y_set)-1):
                            if n == 0:
                                n_b = 0; n_e = Ns_nz;
                            else:
                                n_b = n*Ns_nz; n_e = (n+1)*Ns_nz

                            _trans_prob_aux = trans_prob[ind_prob_pos]
                            _B_LP_AUX = B_fn[n][0](y_set[y_ind])*_trans_prob_aux.reshape(np.size(_trans_prob_aux),1) # Equation 10 Sum of the right hand
                            B_LP[n_b:n_e,0] = _B_LP_AUX.reshape((np.size(_B_LP_AUX)),)

                        t = np.zeros((Ns_nz,1));
                        xi_t0 = np.concatenate((xi_0, t), axis=0)
                   
                        # Since cplexlp solves minimization problems and the problem
                        # is a maximization problem, negate the objective
                        f_cell[a_ind,y_ind-1] = -f_LP.T;
                        A_cell[a_ind,y_ind-1] = csr_matrix(A_LP);
                        B_cell[a_ind,y_ind-1] = B_LP;
                        Aeq_cell[a_ind,y_ind-1] = csr_matrix(Aeq_LP);
                        Beq_cell[a_ind,y_ind-1] = Beq_LP;
                        lb_cell[a_ind,y_ind-1] = lb_LP;
                        ub_cell[a_ind,y_ind-1] = ub_LP(y_set[y_ind]);
                        xi_t0_cell[a_ind,y_ind-1] = xi_t0;

            #cell2mat f_full
            f_full = f_cell[0,0]
            x_x, y_y = np.shape(f_cell)
            for _x in range(x_x):
                for _y in range(y_y):
                    if _x != 0 or _y != 0:
                        f_full = np.concatenate((f_full, f_cell[_x,_y]),axis=0)


            #cell2mat B_full
            B_full = B_cell[0,0]
            x_x, y_y = np.shape(B_cell)
            for _x in range(x_x):
                for _y in range(y_y):
                    if _x != 0 or _y != 0:
                        B_full = np.concatenate((B_full, B_cell[_x,_y]), axis=0)


            #cell2mat lb_cell
            lb_full = lb_cell[0,0];
            x_x, y_y = np.shape(lb_cell)
            for _x in range(x_x):
                for _y in range(y_y):
                    if _x != 0 or _y != 0:
                        lb_full = np.concatenate((lb_full, lb_cell[_x,_y]), axis=0)

            #cell2mat ub_cell
            ub_full = ub_cell[0,0];
            x_x, y_y = np.shape(ub_cell)
            for _x in range(x_x):
                for _y in range(y_y):
                    if _x != 0 or _y != 0:
                        ub_full = np.concatenate((ub_full, ub_cell[_x,_y]), axis=0)

            #cell2mat xi_t0_cell
            xi_t0_full = xi_t0_cell[0,0];
            x_x, y_y = np.shape(xi_t0_cell)
            for _x in range(x_x):
                for _y in range(y_y):
                    if _x != 0 or _y != 0:
                        xi_t0_full = np.concatenate((xi_t0_full, xi_t0_cell[_x,_y]), axis=0)

            #cell2mat A_cell
            print A_cell
            A_concat = A_cell[0,0];
            x_x, y_y = np.shape(A_cell)
            for _x in range(x_x):
                for _y in range(y_y):
                    if _x != 0 or _y != 0:
                        A_concat = scipy.linalg.block_diag(A_concat, A_cell[_x,_y])

            A_full = A_concat[0,0];
            x_x, y_y = np.shape(A_concat)
            for _x in range(x_x):
                for _y in range(y_y):
                    if _x != 0 or _y != 0:
                        print A_concat[_x,_y]
                        A_full = np.concatenate((A_full, A_concat[_x,_y]), axis=0)          
            #A_full = csr_matrix((0,0));
      
            print np.shape(A_full)
            #for nn in range(np.size(A_concat)):
            #    A_full = scipy.linalg.block_diag(A_full,A_concat[:,nn]);
           # print np.shape(A_full)
           # Aeq_concat = Aeq_cell[:];
           # Aeq_full = csr_matrix((0,0));
          #  for nn in range(np.size(Aeq_concat)):
          #      Aeq_full = scipy.linalg.block_diag(Aeq_full,Aeq_concat[:,nn]);
            

    return 0, 0, 0

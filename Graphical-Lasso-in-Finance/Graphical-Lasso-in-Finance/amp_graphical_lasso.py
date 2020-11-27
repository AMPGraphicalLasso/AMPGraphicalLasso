#amp_graphical_lasso.py


import numpy as np
from ampy.NaiveLMMSEVAMPSolver import NaiveLMMSEVAMPSolver
#from ampy.AMPSolver import AMPSolver


def amp_graphical_lasso( X, alpha=0.01, damping_coeff = 1, max_iter = 100, lasso_max_iter=50, convg_threshold=0.1 ):
    """ This function computes the graphical lasso algorithm as outlined in Sparse inverse covariance estimation with the
        graphical lasso (2007).
        
    inputs:
        X: the data matrix, size (nxd)
        alpha: the coefficient of penalization, higher values means more sparseness.
        max_iter: maximum number of iterations
        convg_threshold: Stop the algorithm when the duality gap is below a certain threshold.
        
    
    
    """
    
    if alpha == 0:
        est = cov_estimator(X)
        return est, np.linalg.pinv(est)
    n_features = X.shape[1]

    mle_estimate_ = cov_estimator(X)
    covariance_ = mle_estimate_.copy()
    precision_ = np.linalg.pinv(mle_estimate_)
    indices = np.arange(n_features)
    for i in range( max_iter):
        for n in range( n_features ):
            sub_estimate = covariance_[ indices != n ].T[ indices != n ]
            row = mle_estimate_[ n, indices != n]
            #solve the lasso problem
            amp_obj = NaiveLMMSEVAMPSolver(sub_estimate, row, alpha, damping_coeff)
            #amp_obj = AMPSolver(sub_estimate, row, alpha, damping_coeff)
            amp_obj.solve(max_iteration=lasso_max_iter)
            # _, _, coefs_ = lars_path( sub_estimate, row, Xy = row, Gram = sub_estimate, 
            #                             alpha_min = alpha/(n_features-1.), copy_Gram = True,
            #                             method = "lars")
            coefs_ = amp_obj.x1_hat #just the last please.
            #coefs_ = amp_obj.r
            #coefs_ = coefs_[:,-1] #just the last please.
	          #update the precision matrix.
            precision_[n,n] = 1./( covariance_[n,n] 
                                    - np.dot( covariance_[ indices != n, n ], coefs_  ))
            precision_[indices != n, n] = - precision_[n, n] * coefs_
            precision_[n, indices != n] = - precision_[n, n] * coefs_
            temp_coefs = np.dot( sub_estimate, coefs_)
            covariance_[ n, indices != n] = temp_coefs
            covariance_[ indices!=n, n ] = temp_coefs
	    
        #if test_convergence( old_estimate_, new_estimate_, mle_estimate_, convg_threshold):
        if np.abs( _dual_gap( mle_estimate_, precision_, alpha ) )< convg_threshold:
                break
    else:
        #this triggers if not break command occurs
        #print("Dual Gap is: {}".format(np.abs( _dual_gap( mle_estimate_, precision_, alpha ) )))
        print("The algorithm did not converge. Try increasing the max number of iterations.")
    print("Dual Gap after {} iterations is: {}".format(i+1,np.abs( _dual_gap( mle_estimate_, precision_, alpha ) )))
    return covariance_, precision_
    #i = 0
    # while np.abs( _dual_gap( mle_estimate_, precision_, alpha ) )< convg_threshold:
    #     for n in range( n_features ):
    #         sub_estimate = covariance_[ indices != n ].T[ indices != n ]
    #         row = mle_estimate_[ n, indices != n]
    #         #solve the lasso problem
    #         amp_obj = NaiveLMMSEVAMPSolver(sub_estimate, row, alpha, damping_coeff)
    #         amp_obj.solve(max_iteration=lasso_max_iter)
    #         # _, _, coefs_ = lars_path( sub_estimate, row, Xy = row, Gram = sub_estimate, 
    #         #                             alpha_min = alpha/(n_features-1.), copy_Gram = True,
    #         #                             method = "lars")
    #         coefs_ = amp_obj.x1_hat #just the last please.
    #         # coefs_ = coefs_[:,-1] #just the last please.
	  #         #update the precision matrix.
    #         precision_[n,n] = 1./( covariance_[n,n] 
    #                                 - np.dot( covariance_[ indices != n, n ], coefs_  ))
    #         precision_[indices != n, n] = - precision_[n, n] * coefs_
    #         precision_[n, indices != n] = - precision_[n, n] * coefs_
    #         temp_coefs = np.dot( sub_estimate, coefs_)
    #         covariance_[ n, indices != n] = temp_coefs
    #         covariance_[ indices!=n, n ] = temp_coefs

    #         i= i + 1
	    
        #if test_convergence( old_estimate_, new_estimate_, mle_estimate_, convg_threshold):
        # if np.abs( _dual_gap( mle_estimate_, precision_, alpha ) )< convg_threshold:
        #         break
  
    #this triggers if not break command occurs
    #print("Dual Gap is: {}".format(np.abs( _dual_gap( mle_estimate_, precision_, alpha ) )))
    #print("The algorithm did not converge. Try increasing the max number of iterations.")
    # print("Dual Gap after {} iterations is: {}".format(i+1,np.abs( _dual_gap( mle_estimate_, precision_, alpha ) )))
    # return covariance_, precision_
        
            
def cov_estimator( X ):
    return np.cov(X.T) 
    
    
# def test_convergence( previous_W, new_W, S, t):
#     d = S.shape[0]
#     x = np.abs( previous_W - new_W ).mean()
#     print(x - t*( np.abs(S).sum() + np.abs( S.diagonal() ).sum() )/(d*d-d))
#     if np.abs( previous_W - new_W ).mean() < t*( np.abs(S).sum() + np.abs( S.diagonal() ).sum() )/(d*d-d):
#         return True
#     else:
#         return False

def _dual_gap(emp_cov, precision_, alpha):
    """Expression of the dual gap convergence criterion

    The specific definition is given in Duchi "Projected Subgradient Methods
    for Learning Sparse Gaussians".
    """
    # gap = np.sum(emp_cov * precision_)
    # gap -= precision_.shape[0]
    # gap += alpha * (np.abs(precision_).sum()
    #                 - np.abs(np.diag(precision_)).sum())
    gap = np.trace(np.matmul(emp_cov,precision_)) + alpha * alpha * (np.abs(precision_).sum()- np.abs(np.diag(precision_)).sum()) - precision_.shape[0]
    return gap 


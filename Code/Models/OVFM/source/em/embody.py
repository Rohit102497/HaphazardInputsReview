import numpy as np
from scipy.stats import norm, truncnorm
import warnings
warnings.filterwarnings("error")

def _em_step_body_(args):
    """
    Does a step of the EM algorithm, needed to dereference args to support parallelism
    """
    return _em_step_body(*args)

def _em_step_body(Z, r_lower, r_upper, sigma, num_ord_updates=1):
    """
    Iterate the rows over provided matrix
    """
    num, p = Z.shape
    Z_imp = np.copy(Z)
    C = np.zeros((p,p))
    for i in range(num):
        # try:
        #     c, z_imp, z = _em_step_body_row(Z[i,:], r_lower[i,:], r_upper[i,:], sigma)
        # except:
        #     np.savetxt("Z.txt", Z)
        #     print(Z)
        c, z_imp, z = _em_step_body_row(Z[i,:], r_lower[i,:], r_upper[i,:], sigma)
        Z_imp[i,:] = z_imp
        Z[i,:] = z
        C += c
    return C, Z_imp, Z

def _em_step_body_row(Z_row, r_lower_row, r_upper_row, sigma, num_ord_updates=1):
    """
    The body of the em algorithm for each row
    Returns a new latent row, latent imputed row and C matrix, which, when added
    to the empirical covariance gives the expected covariance
    Args:
        Z_row (array): (potentially missing) latent entries for one data point
        r_lower_row (array): (potentially missing) lower range of ordinal entries for one data point
        r_upper_row (array): (potentially missing) upper range of ordinal entries for one data point
        sigma (matrix): estimate of covariance
        num_ord (int): the number of ordinal columns

    Returns:
        C (matrix): results in the updated covariance when added to the empircal covariance
        Z_imp_row (array): Z_row with latent ordinals updated and missing entries imputed 
        Z_row (array): inpute Z_row with latent ordinals updated
    """
    Z_imp_row = np.copy(Z_row)
    p = Z_imp_row.shape[0]
    num_ord = r_upper_row.shape[0]
    C = np.zeros((p,p))

    obs_indices = np.where(~np.isnan(Z_row))[0]

    missing_indices = np.setdiff1d(np.arange(p), obs_indices)
    ord_in_obs = np.where(obs_indices < num_ord)[0]
    ord_obs_indices = obs_indices[ord_in_obs]
    # obtain correlation sub-matrices
    # obtain submatrices by indexing a "cartesian-product" of index arrays
    sigma_obs_obs = sigma[np.ix_(obs_indices,obs_indices)]
    sigma_obs_missing = sigma[np.ix_(obs_indices, missing_indices)]
    sigma_missing_missing = sigma[np.ix_(missing_indices, missing_indices)]


    if len(missing_indices) > 0:
        tot_matrix = np.concatenate((np.identity(len(sigma_obs_obs)), sigma_obs_missing), axis=1)

        # intermed_matrix = np.linalg.solve(sigma_obs_obs, tot_matrix)
        '''
        The above line of code sometime ran into the error stated below.
        'LinAlgError: Singular matrix'
        
        So, to handle the cases when a matrix might become singlar, we take the approximate solution
        instead of the exact solution.        
        '''
        try:
            intermed_matrix = np.linalg.solve(sigma_obs_obs, tot_matrix)
        except np.linalg.LinAlgError:
            intermed_matrix = np.linalg.lstsq(sigma_obs_obs, tot_matrix, rcond=None)[0]
        
        sigma_obs_obs_inv = intermed_matrix[:, :len(sigma_obs_obs)]
        J_obs_missing = intermed_matrix[:, len(sigma_obs_obs):]
    else:
        # LinAlgError: Singular matrix
        # below code is changes slightly to handle the above warning
        
        # sigma_obs_obs_inv = np.linalg.solve(sigma_obs_obs, np.identity(len(sigma_obs_obs)))
        '''
        The above line of code sometime ran into the error stated below.
        'LinAlgError: Singular matrix'
        
        So, to handle the cases when a matrix might become singlar, we take the approximate solution
        instead of the exact solution.        
        '''
        try:
            sigma_obs_obs_inv = np.linalg.solve(sigma_obs_obs, np.identity(len(sigma_obs_obs)))
        except np.linalg.LinAlgError:
            sigma_obs_obs_inv = np.linalg.lstsq(sigma_obs_obs, np.identity(len(sigma_obs_obs)), rcond=None)[0]

    # initialize vector of variances for observed ordinal dimensions
    var_ordinal = np.zeros(p)

    # OBSERVED ORDINAL ELEMENTS
    # when there is an observed ordinal to be imputed and another observed dimension, impute this ordinal
    if len(obs_indices) >= 2 and len(ord_obs_indices) >= 1:
        for update_iter in range(num_ord_updates):
            # used to efficiently compute conditional mean
            sigma_obs_obs_inv_Z_row = np.dot(sigma_obs_obs_inv, Z_row[obs_indices])
            for ind in range(len(ord_obs_indices)):
                j = obs_indices[ind]
                not_j_in_obs = np.setdiff1d(np.arange(len(obs_indices)),ind) 
                v = sigma_obs_obs_inv[:,ind]
                # new_var_ij = np.asscalar(1.0/v[ind])
                '''
                The above line of code is depriciated. Updated code below
                '''
                new_var_ij = 1.0/v[ind]

                new_mean_ij = Z_row[j] - new_var_ij*sigma_obs_obs_inv_Z_row[ind]

                # To make sure new_var_ij is positive, we take absolute.
                new_var_ij = abs(new_var_ij)

                # print("In embody, new_var_ij", new_var_ij)

                try:
                    mean, var = truncnorm.stats(
                        a=(r_lower_row[j] - new_mean_ij) / np.sqrt(new_var_ij),
                        b=(r_upper_row[j] - new_mean_ij) / np.sqrt(new_var_ij),
                        loc=new_mean_ij,
                        scale=np.sqrt(new_var_ij),
                        moments='mv')
                    if np.isfinite(var):
                        var_ordinal[j] = var
                        if update_iter == num_ord_updates - 1:
                            C[j,j] = C[j,j] + var 
                    if np.isfinite(mean):
                        Z_row[j] = mean
                except OverflowError:
                    print("Truncnorm stats gives overflow error because the range is out of tail. We handle this by assuming that mean and var is infinite.")
                except RuntimeWarning:
                    print("Truncnorm stats gives runtime warning becasue of invalid value encountered in power. We handle this by assuming that mean and var is infinite.")
                

    # MISSING ELEMENTS
    Z_obs = Z_row[obs_indices]
    Z_imp_row[obs_indices] = Z_obs
    if len(missing_indices) > 0:
        Z_imp_row[missing_indices] = np.matmul(J_obs_missing.T,Z_obs) 
        # variance expectation and imputation
        if len(ord_obs_indices) >= 1 and len(obs_indices) >= 2 and np.sum(var_ordinal) > 0: 
            cov_missing_obs_ord = J_obs_missing[ord_in_obs].T * var_ordinal[ord_obs_indices]
            C[np.ix_(missing_indices, ord_obs_indices)] += cov_missing_obs_ord
            C[np.ix_(ord_obs_indices, missing_indices)] += cov_missing_obs_ord.T
            C[np.ix_(missing_indices, missing_indices)] += sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing) \
                                                           + np.matmul(cov_missing_obs_ord, J_obs_missing[ord_in_obs])
        else:
            C[np.ix_(missing_indices, missing_indices)] += sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing)
    return C, Z_imp_row, Z_row
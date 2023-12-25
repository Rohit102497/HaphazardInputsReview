import sys
import numpy as np
from scipy.stats import norm, truncnorm
from statsmodels.distributions.empirical_distribution import ECDF

def sigmoid(x):
    if x > np.log(sys.float_info.max):
        return 1.0
    if x < -np.log(sys.float_info.max):
        return 0.0
    return 1.0 / (1 + np.exp(-x))

def _norm(v):
  sum = float(0)
  for i in range(len(v)):
    sum += v[i]**2
  return sum**(0.5)    
   
# From OVFM Source Code
def em_step_body_row(Z_row: np.array, Z_lower_row: np.array, Z_upper_row: np.array,
                     X_mask: np.array, ord_indices: np.array, sigma: np.array, num_ord_updates: int=2) -> tuple[np.array]:
    """
    The body of the em algorithm for each row
    Returns a new latent row, latent imputed row and C matrix, which, when added
    to the empirical covariance gives the expected covariance
    Args:
        Z_row (array): (potentially missing) latent entries for one data point
        Z_lower_row (array): (potentially missing) lower range of ordinal entries for one data point
        Z_upper_row (array): (potentially missing) upper range of ordinal entries for one data point
        sigma (matrix): estimate of covariance
        num_ord (int): the number of ordinal columns

    Returns:
        C (matrix): results in the updated covariance when added to the empircal covariance
        Z_imp_row (array): Z_row with latent ordinals updated and missing entries imputed 
        Z_row (array): inpute Z_row with latent ordinals updated
    """
    Z_imp_row = np.copy(Z_row)
    p = Z_imp_row.shape[0]
    # num_ord = Z_upper_row.shape[0]
    C = np.zeros((p,p))

    obs_indices = np.where(X_mask)[0]
    missing_indices = np.where(X_mask == 0)[0]

    ord_obs_indices = np.intersect1d(ord_indices, obs_indices)
    # obtain correlation sub-matrices
    # obtain submatrices by indexing a "cartesian-product" of index arrays
    sigma_obs_obs = sigma[np.ix_(obs_indices,obs_indices)]
    sigma_obs_missing = sigma[np.ix_(obs_indices, missing_indices)]
    sigma_missing_missing = sigma[np.ix_(missing_indices, missing_indices)]


    if len(missing_indices) > 0:
        tot_matrix = np.concatenate((np.identity(len(sigma_obs_obs)), sigma_obs_missing), axis=1)
        # print('=============================================================================')
        # print(sigma_obs_obs)
        # print(tot_matrix)
        try:
            intermed_matrix = np.linalg.solve(sigma_obs_obs, tot_matrix)
        except:
            intermed_matrix = np.linalg.lstsq(sigma_obs_obs, tot_matrix, rcond=None)[0]
        sigma_obs_obs_inv = intermed_matrix[:, :len(sigma_obs_obs)]
        J_obs_missing = intermed_matrix[:, len(sigma_obs_obs):]
    else:
        try:
            sigma_obs_obs_inv = np.linalg.solve(sigma_obs_obs, np.identity(len(sigma_obs_obs)))
        except:
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
                # not_j_in_obs = np.setdiff1d(np.arange(len(obs_indices)),ind) 
                v = sigma_obs_obs_inv[:,ind]
                new_var_ij = (1.0/v[ind]).item()

                new_mean_ij = Z_row[j] - new_var_ij*sigma_obs_obs_inv_Z_row[ind]
                mean, var = truncnorm.stats(
                    a=(Z_lower_row[j] - new_mean_ij) / np.sqrt(new_var_ij),
                    b=(Z_upper_row[j] - new_mean_ij) / np.sqrt(new_var_ij),
                    loc=new_mean_ij,
                    scale=np.sqrt(new_var_ij),
                    moments='mv')
                if np.isfinite(var):
                    var_ordinal[j] = var
                    if update_iter == num_ord_updates - 1:
                        C[j,j] = C[j,j] + var 
                if np.isfinite(mean):
                    Z_row[j] = mean

    # MISSING ELEMENTS
    Z_obs = Z_row[obs_indices]
    Z_imp_row[obs_indices] = Z_obs
    if len(missing_indices) > 0:
        Z_imp_row[missing_indices] = np.matmul(J_obs_missing.T,Z_obs) 
        # variance expectation and imputation
        if len(ord_obs_indices) >= 1 and len(obs_indices) >= 2 and np.sum(var_ordinal) > 0: 
            cov_missing_obs_ord = J_obs_missing[ord_obs_indices].T * var_ordinal[ord_obs_indices]
            C[np.ix_(missing_indices, ord_obs_indices)] += cov_missing_obs_ord
            C[np.ix_(ord_obs_indices, missing_indices)] += cov_missing_obs_ord.T
            C[np.ix_(missing_indices, missing_indices)] += sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing) \
                                                           + np.matmul(cov_missing_obs_ord, J_obs_missing[ord_obs_indices])
        else:
            C[np.ix_(missing_indices, missing_indices)] += sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing)
    return C, Z_imp_row

# From OVFM Source Code
def project_to_correlation(covariance: np.array) -> np.array:
    """
    Projects a covariance to a correlation matrix, normalizing it's diagonal entries
    Args:
        covariance (matrix): a covariance matrix
    Returns:
        correlation (matrix): the covariance matrix projected to a correlation matrix
    """
    D = np.diagonal(covariance)
    D = np.where(D, D, sys.float_info.min)
    D_neg_half = np.sign(D) * 1.0/np.sqrt(np.abs(D))
    covariance *= D_neg_half
    return covariance.T * D_neg_half

def get_cont_ord_indices(X: np.array, X_mask: np.array) -> tuple[list]:
    max_ord=14
    cont_indices = []
    ord_indices = []
    for i in range(X.shape[-1]):
        col_non_masked = X[X_mask[:, i].astype(bool), i]
        if len(np.unique(col_non_masked) > max_ord):
            cont_indices.append(i)
        else:
            ord_indices.append(i)
    return cont_indices, ord_indices

class TransformFunction:
    def __init__(self, cont_indices: list, ord_indices: list, X_batch: np.array=None, X_batch_mask: np.array=None, B: int=20):
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        self.window = np.ones((B, X_batch.shape[1])) * np.nan
        self.window_size = B

        if X_batch is not None and X_batch_mask is not None:
            self.update(X_batch, X_batch_mask)
    
    def update(self, X_batch, X_batch_mask):
        if len(X_batch.shape) == 1:
            X_batch = X_batch.reshape(1, -1)
            X_batch_mask = X_batch_mask.reshape(1, -1)

        if np.isnan(self.window[0, 0]):
            self.init_window(X_batch, X_batch_mask)
        
        for i in range(X_batch.shape[0]):
            for j in range(X_batch.shape[1]):
                if X_batch_mask[i, j]:
                    self.window[i, j] = X_batch[i, j]
        
        return
    
    def init_window(self, X_batch, X_batch_mask):
        # Initalization

        # Continuous columns: normal initialization
        mean = np.mean(X_batch[:, self.cont_indices] * X_batch_mask[:, self.cont_indices], axis = 0) # returns an array of shape (len(self.cont_indices), )
        std = np.std(X_batch[:, self.cont_indices] * X_batch_mask[:, self.cont_indices], axis = 0) # returns an array of shape (len(self.cont_indices), )
        '''In OVFM source code, they're not taking column wise mean and std, but of the whole matrix (see code below)'''
        self.window[:, self.cont_indices] = np.random.normal(mean, std, size=(self.window_size, len(self.cont_indices)))


        # Ordinal Columns: Uniform Initialization
        for i in self.ord_indices:
            col = X_batch[:, i]
            col_mask = X_batch_mask[:, i]
            col = col[col_mask.astype(bool)]

            min_ord = np.min(col)
            max_ord = np.max(col)

            self.window[:, i] = np.random.randint(min_ord, max_ord+1, self.window_size)
        
        return
 
    def get_cont_latent(self, X, X_mask) -> np.array:
        """
        Obtain the latent continuous values corresponding to X_batch 
        """
        Z_cont = np.empty(X.shape)
        Z_cont[:] = np.nan
        not_missing = np.where(X_mask)[0]
        for i in self.cont_indices:
            if i in not_missing:
                """
                Return the latent variables corresponding to the continuous entries of 
                self.X. Estimates the CDF columnwise with the empyrical CDF
                """
                ecdf = ECDF(self.window[:, i])
                l = len(self.window[:, i])
                q = (l / (l + 0.1)) * ecdf(X[i])
                q = l/(l+1)/2 if q==0 else q
                if q==0:
                    print("In get_cont_latent, 0 quantile appears")
                    return
                Z_cont[i] = norm.ppf(q)

        return Z_cont[self.cont_indices]

    def get_ord_latent(self, X, X_mask) -> tuple[np.array]:
        """
        Obtain the latent ordinal values corresponding to X_batch
        """
        Z_ord_lower = np.empty(X.shape)
        Z_ord_lower[:] = np.nan
        Z_ord_upper = np.empty(X.shape)
        Z_ord_upper[:] = np.nan
        not_missing = np.where(X_mask)[0]
        for i in self.ord_indices:
            if i in not_missing:
                """
                get the cdf at each point in X_batch
                """
                ecdf = ECDF(self.window[:, i])
                unique = np.unique(self.window[:, i])
                if unique.shape[0] > 1:
                    threshold = np.min(np.abs(unique[1:] - unique[:-1]))/2.0
                    z_lower_obs = norm.ppf(ecdf(X[i] - threshold))
                    z_upper_obs = norm.ppf(ecdf(X[i] + threshold))
                else:
                    z_upper_obs = np.inf
                    z_lower_obs = -np.inf

                Z_ord_lower[i] = z_lower_obs
                Z_ord_upper[i] = z_upper_obs

        return Z_ord_lower[self.ord_indices], Z_ord_upper[self.ord_indices]

    def get_cont_observed(self, Z, X_mask, X=None) -> np.array:
        """
        Transform the latent continous variables in Z_batch into corresponding observations
        """
        if X is not None:
            X_cont_imp = np.copy(X)
        else:
            X_cont_imp = np.zeros(Z.shape) * np.nan

        missing = np.where(X_mask == 0)[0]
        for i in self.cont_indices:
            if i in missing:
                """
                Applies marginal scaling to convert the latent entries in Z corresponding
                to continuous entries to the corresponding imputed oberserved value
                """
                if (len(missing) > 0):
                    quantiles = norm.cdf(Z[i])
                    if np.isnan(quantiles):
                        quantiles = 1.0
                    X_cont_imp[i] = np.quantile(self.window[:, i], quantiles)
        
        return X_cont_imp[self.cont_indices]
    
    def get_ord_observed(self, Z, X_mask, X=None, DECIMAL_PRECISION = 3) -> np.array:
        """
        Transform the latent ordinal variables in Z_batch into corresponding observations
        """
        if X is not None:
            X_ord_imp = np.copy(X)
        else:
            X_ord_imp = np.zeros(Z.shape) * np.nan
        
        for i in self.ord_indices:
            missing = np.where(X_mask[:, i] == 0)[0]
            if (len(missing) > 0):
                """
                Gets the inverse CDF of Q_batch
                returns: the Q_batch quantiles of the ordinals seen thus far
                """
                n = len(self.window)
                x = norm.cdf(Z[missing, i])
                # round to avoid numerical errors in ceiling function
                quantile_indices = np.ceil(np.round_((n + 1) * x - 1, DECIMAL_PRECISION))
                quantile_indices = np.clip(quantile_indices, a_min=0,a_max=n-1).astype(int)
                sort = np.sort(self.window[:, i])
                X_ord_imp[missing, i] =  sort[quantile_indices]

        return X_ord_imp[self.ord_indices]
    
    def init_Z_ord(self, Z_ord_lower, Z_ord_upper) -> np.array:
        """
        Initializes the observed latent ordinal values by sampling from a standard
        Gaussian truncated to the inveral of Z_ord_lower, Z_ord_upper

        Args:
            Z_ord_lower (matrix): lower range for ordinals
            Z_ord_upper (matrix): upper range for ordinals

        Returns:
            Z_ord (range): Samples drawn from gaussian truncated between Z_ord_lower and Z_ord_upper
        """
        Z_ord = np.empty(Z_ord_lower.shape)
        Z_ord[:] = np.nan

        k = Z_ord.shape[-1]
        obs_indices = ~np.isnan(Z_ord_lower)

        u_lower = np.copy(Z_ord_lower)
        u_lower[obs_indices] = norm.cdf(Z_ord_lower[obs_indices])
        u_upper = np.copy(Z_ord_upper)
        u_upper[obs_indices] = norm.cdf(Z_ord_upper[obs_indices])

        for j in range(k):
            if not np.isnan(Z_ord_upper[j]) and u_upper[j] > 0 and u_lower[j]<1:
                u_sample = np.random.uniform(u_lower[j],u_upper[j])
                Z_ord[j] = norm.ppf(u_sample)
        return Z_ord
    
    def get_latent(self, X, X_mask) -> tuple[np.array]:
        """
        Obtain the latent (continuous and ordinal) values corresponding to X_batch 
        """
        Z = np.empty(X.shape)

        Z_cont = self.get_cont_latent(X, X_mask)
        Z_ord_lower, Z_ord_upper = self.get_ord_latent(X, X_mask)

        Z[self.cont_indices] = Z_cont
        Z[self.ord_indices] = self.init_Z_ord(Z_ord_lower, Z_ord_upper)

        return Z, Z_ord_lower, Z_ord_upper

    def get_observed(self, Z, X_mask, X=None) -> np.array:
        """
        Transform the latent (continous and ordinal) variables in Z_batch into corresponding observations
        """
        X = np.empty(Z.shape)
        
        X_cont = self.get_cont_observed(Z, X_mask, X)
        X_ord = self.get_ord_observed(Z, X_mask, X)

        X[self.cont_indices] = X_cont
        X[self.ord_indices] = X_ord

        return X

class Copula:
    def __init__(self, cont_indices: list, ord_indices: list,
                 X_batch: np.array, X_batch_mask: np.array, B: int=20):
        self.TransformFunction = TransformFunction(cont_indices, ord_indices, X_batch, X_batch_mask, B)
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        self.sigma = np.identity(X_batch.shape[1])
        self.batch_size = X_batch.shape[0]

    def partial_fit(self, X: np.array, X_mask: np.array, decay_coef: float=0.5):
        Z, Z_lower, Z_upper = self.TransformFunction.get_latent(X, X_mask)
        C, Z_imp= em_step_body_row(Z, Z_lower, Z_upper, X_mask, self.ord_indices, self.sigma)
        X_imp = self.TransformFunction.get_observed(Z_imp, X_mask, X)

        C = C/self.batch_size
        sigma = np.cov(Z_imp, rowvar=False) + C
        sigma = project_to_correlation(sigma)
        self.sigma = sigma*decay_coef + (1 - decay_coef)*self.sigma

        self.TransformFunction.update(X, X_mask)

        return X_imp, Z_imp

class Classifier:
    def __init__(self, num_feat: int, alpha: float=0.5, T: int=None, lr: float=0.01, c: float=0.5):
        self.W_obs = np.ones(num_feat) / np.sqrt(num_feat)
        self.W_lat = np.ones(num_feat) / np.sqrt(num_feat)
        self.lr = lr
        self.alpha = alpha
        self.R_obs = 0
        self.R_lat = 0
        self.T = T
        self.i = 0 # to keep track of number of updates
        self.tau = 2 * np.sqrt(2*np.log(2)/self.T)
        self.c = c
    
    def loss(self,y_true, y_pred):
        if y_true == 1:
            y_pred = sys.float_info.min if y_pred == 0 else y_pred
            return -np.log(y_pred)
        if y_pred == 1:
            return -np.log(sys.float_info.min)
        return -np.log(1-y_pred)
    
    def predict(self, X: np.array, Z: np.array) -> np.float64:
        y_logit = self.alpha*np.dot(X, self.W_obs) + (1-self.alpha)*np.dot(Z, self.W_lat)
        return y_logit.item()
    
    def update(self, y: float, y_logit: np.float64, X: np.array, Z: np.array):
        # Update weights and bias using BCE loss gradient
        error = y_logit - y
        self.W_obs -= self.lr * error * X
        self.W_lat -= self.lr * error * Z

        self.i+=1

        if self.i == self.T:
            self.i=0
            self.R_obs = 0
            self.R_lat = 0
        
        self.R_obs += self.loss(y, sigmoid(np.dot(X, self.W_obs)))
        self.R_lat += self.loss(y, sigmoid(np.dot(Z, self.W_lat)))

        term1 = np.exp(-self.tau*self.R_obs)
        term2 = np.exp(-self.tau*self.R_lat)

        self.alpha =  term1 / (term1+term2) if term1!=0 else 0

        W_obs_norm = _norm(self.W_obs)
        W_lat_norm = _norm(self.W_lat)

        self.W_obs = np.minimum(1, self.c/W_obs_norm)*self.W_obs if W_obs_norm != 0 else 1
        self.W_lat = np.minimum(1, self.c/W_lat_norm)*self.W_lat if W_lat_norm != 0 else 1

class OVFM:
    def __init__(self, c: float=None, all_cont: bool=False, lr: float=0.01, B: int=20,
                 X: np.array=None, X_mask: np.array=None):
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            X_mask = X_mask.reshape(1, -1)
        
        self.B = B          # Buffer Size
        self.c = c          # Sparcity Factor
        # self.e = e        # endpoint (This was used in the paper, but not in the gaussin copula implementation of code)

        if all_cont or X.shape[0] == 1:
            cont_indices = np.where(np.sum(X_mask, axis=0))[0]
            ord_indices = []
        else:
            cont_indices, ord_indices = get_cont_ord_indices(X, X_mask)
        
        self.Copula = Copula(cont_indices, ord_indices, X, X_mask, B)
        self.Classifier = Classifier(X.shape[1], T = self.B, lr=lr)

    def predict(self, X: np.array, X_mask: np.array) -> tuple[np.array]:
        X_imp, Z_imp = self.Copula.partial_fit(X, X_mask)
        y_logit = self.Classifier.predict(X_imp, Z_imp)
        y_pred = int(y_logit > 0)

        return y_pred, y_logit

    def update(self, y: float, y_logit: np.float64, X_imp: np.array, Z_imp: np.array):
        self.Classifier.update(y, y_logit, X_imp, Z_imp)

    def partial_fit(self, X: np.array, X_mask: np.array, Y: np.array) -> tuple[np.array]:
        X_imp, Z_imp = self.Copula.partial_fit(X, X_mask)
        y_logit = self.Classifier.predict(X_imp, Z_imp)
        y_pred = int(y_logit > 0)

        self.update(Y, y_logit, X_imp, Z_imp)

        return y_pred, y_logit
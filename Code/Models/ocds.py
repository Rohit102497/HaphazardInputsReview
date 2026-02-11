"""
haphazard/models/model_zoo/ocds/ocds.py
---------------------------------------
Implementation of OCDS (GLSC) algorithm.

Papers
    GLSC: https://ieeexplore.ieee.org/document/9066871
    OCDS: https://www.ijcai.org/proceedings/2019/346
"""

import numpy as np
from numpy.typing import NDArray


class OCDS:
    """
    Class implementing the GLSC algorithm.
    """
    def __init__(
            self, 
            alpha: float, 
            lamda: float, 
            beta1: float, 
            gamma: float, 
            num_features: int ,
            tau : float|None = None,
            eps: float = 1e-12,
            clip_val: float = 1e3,        # [GUARD] clip range for exploding values
            min_norm: float = 1e-8        # [GUARD] prevent vanishing norms
        ) -> None:
        """
        Initialize hyperparameters.
        
        Parameters
        ----------
        alpha : float
            Coefficient of reconstruction loss.
        lamda : float
            Coefficient of model regularizer.
        beta1 : float
            β2, from `β1 = λβ, and β2 = 2λ(1 - β)`
            where, β is the tradeoff parameter between weight regularization and graph regularization.
            Used to calculate β2, as `β2 = 2λ - 2β1`.
        gamma : float
            Truncation ratio for the weight vector (classifier).
        tau : float|None
            Learning rate for linear classifier (`w`).
        num_features : int
            Total (max) number of features to be observed, only used for ease of coding.
        eps : float
            Small positive value to avoid computational instability (like division by zero).
        clip_val : float
            Clipping value to prevent exploding gradients
        min_norms : float
            Clipping value to prevent vanishing norms
        """

        # Assign hyperparameters
        self.alpha = alpha
        self.lamda = lamda
        self.beta1 = beta1
        self.beta2 = 2*lamda - 2*beta1
        self.gamma = gamma
        self.tau = tau
        self.num_features = num_features
        self.eps = eps
        self.clip_val = clip_val
        self.min_norm = min_norm

        if self.tau is None:
            print(f"Found tau={tau!r}. Will use tau=np.sqrt(1.0/t) as a varied step size.")

        # Initialize parameters
        self.w = np.zeros(self.num_features)
        self.Ut = set()
        self.G = np.random.random((self.num_features, self.num_features))  # pre-initialized

        self.T: int = 0
        self.p = 0.5
        self.L_obs = 0.0
        self.L_rec = 0.0


    def loss(self, y: int, y_hat: float) -> float:
        """
        Numerically stable logistic loss.
        """
        yz = y * y_hat
        # [GUARD] stable softplus form, safe for large yz
        stable = np.maximum(0.0, -yz) + np.log1p(np.exp(-np.abs(yz)))
        return stable / np.log(2.0)  # convert to log base 2 as in paper

    
    
    def laplacian(self) -> NDArray[np.float64]:
        A = 0.5 * (self.G + self.G.T)
        deg = np.sum(A, axis=1)
        return np.diag(deg) - A
    
    def partial_fit(
            self, 
            x: NDArray[np.float64], 
            mask: NDArray[np.bool_], 
            y: NDArray[np.int64] | int
        ) -> tuple[int, float]:
        """
        Partially fit (predict and update) on a single instance.

        Parameters
        ----------
        x, NDArray[np.float64]:
            Haphazard feature vector.
        mask, NDArray[np.bool_]:
            Mask indicating observed features.
        y, NDArray[np.int64] | int:
            Ground truth label.

        Returns
        -------
        pred, int:
            Predicted class.
        logit, float:
            Un-normalized prediction logit.
        """
        x = np.asarray(x, dtype=float).ravel()
        mask = np.asarray(mask, dtype=bool).ravel()
        y = int(y.squeeze())
        y = {0:-1, 1:1}[y]  # Convert binary labels from {0, 1} to {-1, 1}

        # ==== Algorithm 1: GLSC Algorithm ====
        # dt: number of observed features in the instance
        # d : total number of observevable features (here self.num_features)

        # --- 1. Increment iteration count ---
        self.T += 1


        # --- 2. Update universal feature-space ---
        self.Ut = self.Ut.union(set(np.where(mask)[0]))


        # --- 3. Retrive Gr from self.G ---
        dt = sum(mask.astype(int))
        x_obs = x[mask]                     # shape (dt,)
        Gr = self.G[mask, :]                # shape (dt, d)

        # --- Reconstruct missing features ---
        psi = (1.0 / dt) * (Gr.T @ x_obs)   # shape (d,)
        x_rec = psi[~mask]                  # shape (d-dt,)


        # --- 4. Predict score and label ---
        w_obs = self.w[mask]                # shape (dt,)
        score_obs = float(np.dot(x_obs, w_obs))

        
        w_rec = self.w[~mask]               # shape (d-dt,)
        score_rec = float(np.dot(x_rec, w_rec))

        y_hat = self.p * score_obs + (1-self.p) * score_rec
        logit = float(y_hat)
        label = 1 if logit>0 else -1


        # --- 5. Update running losses ---
        self.L_obs += self.loss(y, score_obs)
        self.L_rec += self.loss(y, score_rec)


        # --- 6. Update self.p ---
        lnT = max(np.log(self.T), self.eps)                 # [GUARD] avoid log(0)
        eta = 8.0 * np.sqrt(1.0 / lnT)
        exp_obs = np.exp(np.clip(-eta * self.L_obs, -50, 50))  # [GUARD] clamp exp input
        exp_rec = np.exp(np.clip(-eta * self.L_rec, -50, 50))  # [GUARD] clamp exp input
        denominator = exp_obs + exp_rec + self.eps
        self.p = float(exp_obs / denominator)


        # --- 7. Update self.w and self.G ---
        # Simplification:
        # residual  = y - (1/dt) * w.T @ (IG).T @ x_obs
        #           = y - w.T [(1/dt) * Gr.T @ x_obs]
        #           = y - w.T @ psi
        residual = y - float(self.w.T @ psi)  # scalar

        # --- Update self.w ---
        grad_w1 = (-2.0 / dt) * residual * psi             # shape (d,)
        grad_w2 = self.lamda * np.sign(self.w)      # shape (d,) [subgradient of lambda * ||w||_1]
        L = self.laplacian()
        grad_w3 = self.beta2* ((L + L.T) @ self.w)  # shape (d,)
        
        grad_w = (grad_w1 + grad_w2 + grad_w3)      # shape (d,)

        # [GUARD] gradient clipping to prevent exploding updates
        grad_w = np.clip(grad_w, -self.clip_val, self.clip_val)
        tau = np.sqrt(1.0/self.T) if self.tau is None else self.tau
        self.w = self.w - tau * grad_w

        # [GUARD] weight norm stabilization (prevent NaN/Inf or vanishing)
        w_norm = np.linalg.norm(self.w)
        if not np.isfinite(w_norm) or w_norm < self.min_norm:
            self.w = np.sign(self.w) * self.min_norm

        # --- Update self.G ---
        I = np.eye(self.num_features, self.num_features)[mask, :]                           # shape (dt, d)
        grad_G1 = (-2.0 / dt) * residual * np.outer(I.T @ x_obs, self.w.T)                  # shape (d, d)
        grad_G2 = (-2.0 * self.alpha / dt) * np.outer(I.T @ x_obs, (x_obs - I @ psi).T) @ I  # shape (d, d)
        grad_G = grad_G1 + grad_G2

        # [GUARD] gradient clipping for G
        grad_G = np.clip(grad_G, -self.clip_val, self.clip_val)

        # split positive and negative parts elementwise
        gradG_pos = np.maximum(grad_G, 0.0)
        gradG_neg = np.maximum(-grad_G, 0.0)

        ratio = (gradG_neg + self.eps) / (gradG_pos + self.eps)
        ratio = np.clip(ratio, 1e-3, 1e3)  # [GUARD] bound multiplicative ratio

        self.G *= ratio
        np.clip(self.G, 0.0, 1e6, out=self.G)  # [GUARD] cap extreme positive values

        # [GUARD] prevent G collapse (row normalization if tiny)
        row_norms = np.linalg.norm(self.G, axis=1, keepdims=True) + self.eps
        tiny_rows = row_norms < self.min_norm
        if np.any(tiny_rows):
            self.G[tiny_rows[:, 0]] += self.min_norm  # nudge to avoid all-zero rows


        # --- 8. Truncate self.w based on gamma ---
        if 0.0 < self.gamma < 1.0 and self.w.size > 0:
            k = max(1, int(np.ceil(self.gamma * self.num_features)))
            absw = np.abs(self.w)
            if k < self.num_features:
                thr = np.partition(absw, -k)[-k]
                self.w[absw < thr] = 0.0
        
        pred = {-1:0, 1:1}[label]  # Convert binary labels from {-1, 1} back to {0, 1}
        return pred, logit

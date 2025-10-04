
import numpy as np
import pandas as pd

class CoxPH:
    """
    Cox Proportional Hazards (vraisemblance partielle) avec:
    - Newton-Raphson + backtracking line search
    - Pénalisation L2 (optionnelle, ridge)
    - Breslow pour ties
    """
    def __init__(self, l2=0.0, max_iter=100, tol=1e-6, verbose=False):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = verbose
        self.beta_ = None
        self.baseline_cumhaz_ = None  # DataFrame: columns = ['time', 'H0']
        self.event_times_ = None
        self.mean_X_ = None  # pour centrer (optionnel)
        self.std_X_ = None   # pour standardiser (optionnel)

    def _loglik_grad_hess(self, T, E, X, beta):
        n, p = X.shape
        XB = X @ beta
        exp_XB = np.exp(XB)

        order = np.argsort(T, kind="mergesort")
        T_ord, E_ord, X_ord = T[order], E[order], X[order, :]
        XB_ord = XB[order]
        exp_XB_ord = exp_XB[order]

        S0_rev = np.cumsum(exp_XB_ord[::-1])[::-1]
        S1_rev = np.cumsum((X_ord * exp_XB_ord[:, None])[::-1], axis=0)[::-1]

        first_index_by_time = {t: np.searchsorted(T_ord, t, side='left') for t in np.unique(T_ord)}

        loglik = 0.0
        grad = np.zeros(p)
        H = np.zeros((p, p))

        i_evt = np.where(E_ord == 1)[0]
        from collections import defaultdict
        groups = defaultdict(list)
        for idx in i_evt:
            groups[T_ord[idx]].append(idx)

        for t_j, idxs in groups.items():
            d_j = len(idxs)
            k = first_index_by_time[t_j]
            S0 = S0_rev[k]
            S1 = S1_rev[k]

            sum_XB_events = XB_ord[idxs].sum()
            sum_X_events = X_ord[idxs, :].sum(axis=0)

            loglik += sum_XB_events - d_j * np.log(S0)
            grad += sum_X_events - d_j * (S1 / S0)

            mu = S1 / S0
            X_risk = X_ord[k:, :]
            w_risk = exp_XB_ord[k:]
            X_weighted = X_risk * w_risk[:, None]
            E_xx = (X_risk.T @ X_weighted) / S0
            Cov = E_xx - np.outer(mu, mu)
            H -= d_j * Cov

        if self.l2 > 0:
            loglik -= 0.5 * self.l2 * np.dot(beta, beta)
            grad -= self.l2 * beta
            H -= self.l2 * np.eye(p)

        return loglik, grad, H

    def fit(self, X, T, E, standardize=True):
        """
        X : array (n, p), T : temps (n,), E : événement (1) / censuré (0)
        """
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float)
        E = np.asarray(E, dtype=int)
        n, p = X.shape

        if standardize:
            self.mean_X_ = X.mean(axis=0)
            self.std_X_ = X.std(axis=0)
            self.std_X_[self.std_X_ == 0] = 1.0
            Xs = (X - self.mean_X_) / self.std_X_
        else:
            self.mean_X_ = np.zeros(p)
            self.std_X_ = np.ones(p)
            Xs = X

        beta = np.zeros(p)
        prev_ll = -np.inf

        for _ in range(self.max_iter):
            ll, g, H = self._loglik_grad_hess(T, E, Xs, beta)
            grad_norm = np.linalg.norm(g, ord=np.inf)
            if np.isfinite(prev_ll) and abs(ll - prev_ll) < self.tol and grad_norm < 1e-4:
                break
            prev_ll = ll

            try:
                step = np.linalg.solve(-H, g)
            except np.linalg.LinAlgError:
                step = g / max(1.0, np.linalg.norm(g))

            t = 1.0
            c = 1e-4
            while t > 1e-8:
                beta_new = beta + t * step
                ll_new, _, _ = self._loglik_grad_hess(T, E, Xs, beta_new)
                if ll_new >= ll + c * t * np.dot(g, step):
                    beta = beta_new
                    break
                t *= 0.5
            else:
                beta = beta_new
                break

        self.beta_ = beta
        self._compute_baseline_cumhaz(T, E, Xs)
        return self

    def _compute_baseline_cumhaz(self, T, E, Xs):
        order = np.argsort(T, kind="mergesort")
        T_ord, E_ord, X_ord = T[order], E[order], Xs[order, :]
        XB_ord = X_ord @ self.beta_
        exp_XB_ord = np.exp(XB_ord)

        S0_rev = np.cumsum(exp_XB_ord[::-1])[::-1]
        unique_event_times = np.unique(T_ord[E_ord == 1])

        H0 = []
        cum = 0.0
        first_index_by_time = {t: np.searchsorted(T_ord, t, side='left') for t in np.unique(T_ord)}
        from collections import defaultdict
        groups = defaultdict(list)
        for idx in np.where(E_ord == 1)[0]:
            groups[T_ord[idx]].append(idx)

        for t in sorted(unique_event_times):
            d_j = len(groups[t])
            k = first_index_by_time[t]
            S0 = S0_rev[k]
            increment = d_j / S0
            cum += increment
            H0.append((t, cum))

        self.baseline_cumhaz_ = pd.DataFrame(H0, columns=["time", "H0"])
        self.event_times_ = self.baseline_cumhaz_["time"].values

    def predict_log_partial_hazard(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_X_) / self.std_X_
        return Xs @ self.beta_

    def predict_partial_hazard(self, X):
        return np.exp(self.predict_log_partial_hazard(X))

    def predict_survival_function(self, X, times=None):
        if self.baseline_cumhaz_ is None:
            raise RuntimeError("Fit le modèle avant de prédire la survie.")
        H0_df = self.baseline_cumhaz_
        if times is None:
            times = H0_df["time"].values
        times = np.asarray(times, dtype=float)
        H0_at_t = np.interp(times, H0_df["time"].values, H0_df["H0"].values, left=0.0, right=H0_df["H0"].values[-1])
        lp = self.predict_log_partial_hazard(X)
        hr = np.exp(lp)
        S = np.exp(-np.outer(H0_at_t, hr))
        cols = [f"obs_{i}" for i in range(S.shape[1])]
        return pd.DataFrame(S, index=times, columns=cols)

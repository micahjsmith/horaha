import logging
from collections import defaultdict

import funcy
import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.stats
from tqdm import tqdm

logger = logging.getLogger('horaha')


class LatentPositionModel:
    def __init__(self, k):
        self.k = k

    def ηij(self, yij, α, zi, zj):
        raise NotImplementedError

    def logpa(self, α):
        return np.asscalar(np.log(self.pa(α)))

    def logpz(self, Z):
        assert Z.shape[0] == self.k
        return np.asscalar(np.log(self.pz(Z)))

    def logprior(self, α, Z):
        return np.asscalar(self.logpa(α) + self.logpz(Z))

    def loglikelihood_ij(self, yij, α, zi, zj):
        '''Compute the log likelihood for actors i and j.

        The log likelihood (Eq. 4) is given as

            p(y_{i,j} = 1 | \alpha, z_i, z_j) =
                \eta_{i,j} * y_{i,j} - log(1 + e^{\eta_{i,j}})

        where the \etas are model-specific given as

            \eta_{i,j} = logit p(y_{i,j} = 1 | \cdot)

        Note: assumes y_ij \in {0, 1}.

        '''
        ηij = self.ηij(α, zi, zj)
        log1pex = np.logaddexp(0, ηij)
        return ηij * yij - log1pex

    def loglikelihood(self, Y, α, Z):
        '''Compute the log likelihood of the data.

        The log likelihood is given as

            p(Y | \alpha, Z) =
                \sum_{i \ne j}^{n} log p(y_{i,j} | \alpha, z_i, z_j)

        '''
        n = Y.shape[0]
        assert n == Y.shape[1]

        result = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                result += self.loglikelihood_ij(Y[i, j], α, Z[:, i], Z[:, j])

        return result

    def logposterior(self, Y, α, Z):
        return self.logprior(α, Z) + self.loglikelihood(Y, α, Z)

    def make_obj(self, Y, kind='logposterior'):
        if kind not in ['likelihood', 'loglikelihood',
                        'posterior', 'logposterior']:
            raise ValueError

        g = getattr(self, kind)
        n = Y.shape[0]

        def f(θ):
            return -g(Y, θ[0], θ[1:].reshape(-1, n))

        return f

    def optimize(self, Y, α0, Z0, kind):
        n = Y.shape[0]
        f = self.make_obj(Y, kind=kind)

        # create optimization starting point
        if α0 is None:
            α0 = np.zeros(1)
        else:
            assert np.array(α0).shape == (1,)
        if Z0 is None:
            Z0 = np.zeros(n * self.k)
        else:
            if Z0.ndim > 1:
                Z0 = Z0.ravel()
            assert np.array(Z0).shape == (self.k * n,)
        x0 = np.concatenate([α0, Z0])

        result = scipy.optimize.minimize(f, x0)
        return result

    def mle(self, Y, α0=None, Z0=None):
        logger.info('Beginning maximum likelihood estimation...')
        result = self.optimize(Y, α0, Z0, kind='loglikelihood')
        α, Z = result.x[0], result.x[1:].reshape(self.k, -1)
        logger.info('Beginning maximum likelihood estimation...DONE')
        return α, Z

    def map(self, Y, α0=None, Z0=None):
        logger.info('Beginning maximum a posteriori estimation...')
        result = self.optimize(Y, α0, Z0, kind='logposterior')
        α, Z = result.x[0], result.x[1:].reshape(self.k, -1)
        logger.info('Beginning maximum a posteriori estimation...DONE')
        return α, Z

    def _compute_procrustean_transformation(self, Z, Z0):
        # first, center Z at the origin. (assumes Z0 is already centered.)
        # computes
        #    Z^{*} = Z_0 Z^{′} (Z Z_0^{′} Z_0 Z^{′})^{-1/2} Z
        Z = Z - np.mean(Z, axis=1).reshape(-1, 1)
        Z_star = (
            Z0
            .dot(Z.T)
            .dot(
                scipy.linalg.sqrtm(
                    np.linalg.inv(
                        Z
                        .dot(Z0.T)
                        .dot(Z0)
                        .dot(Z.T)
                    )
                )
            )
            .dot(Z)
        )
        return Z_star

    def qZ(self, Z):
        mean = Z.ravel()
        cov = self.σ_q_zi
        r = scipy.stats.multivariate_normal.rvs(mean=mean, cov=cov)
        r = r.reshape(Z.shape)
        return r

    def qα(self, α):
        σ = self.σ_q_α
        return scipy.stats.norm.rvs(loc=α, scale=σ)

    def sample(self, Y, K=2.5e6, h=2e3, α_hat=None, Z_hat=None):
        K, h = int(K), int(h)
        result = defaultdict(list)

        try:
            # compute MLE and center it at origin
            if α_hat is None or Z_hat is None:
                α_hat, Z_hat = self.mle(Y)
            Z_hat = Z_hat - np.mean(Z_hat, axis=1).reshape(-1, 1)

            result['α_hat'] = α_hat
            result['Z_hat'] = Z_hat

            Z_k = Z_hat.copy()
            α_k = α_hat.copy()

            def update_Z(α_k, Z_k, detail=False):
                Z_dag = self.qZ(Z_k)

                # compute acceptance probability
                n1 = self.loglikelihood(Y, α_k, Z_dag)
                d1 = self.loglikelihood(Y, α_k, Z_k)
                n2 = self.logpz(Z_dag)
                d2 = self.logpz(Z_k)
                p = np.exp(n1 + n2 - d1 - d2)
                a = min(1, p)

                # accept/reject step
                u = scipy.stats.uniform.rvs()
                accepted = u < a
                if accepted:
                    Z_k = Z_dag
                else:
                    pass

                if detail:
                    _detail = {
                        'Z_dag': Z_dag, 'n1': n1, 'd1': d1, 'n2': n2,
                        'd2': d2, 'p': p, 'a': a, 'u': u, 'accepted': accepted,
                    }
                else:
                    _detail = None

                return Z_k, _detail

            def update_α(α_k, Z_k, detail=False):
                α_dag = self.qα(α_k)

                # compute acceptance probability
                n1 = self.loglikelihood(Y, α_dag, Z_k)
                d1 = self.loglikelihood(Y, α_k, Z_k)
                n2 = self.logpa(α_dag)
                d2 = self.logpa(α_k)
                p = np.exp(n1 + n2 - d1 - d2)
                a = min(1, p)

                # accept/reject step
                u = scipy.stats.uniform.rvs()
                accepted = u < a
                if accepted:
                    α_k = α_dag
                else:
                    pass

                if detail:
                    _detail = {
                        'α_dag': α_dag, 'n1': n1, 'd1': d1, 'n2': n2,
                        'd2': d2, 'p': p, 'a': a, 'u': u, 'accepted': accepted,
                    }
                else:
                    _detail = None

                return α_k, _detail

            logger.info('Sampling...')
            for k in tqdm(range(K)):
                # record results every h iterations
                record_iter = k % h == 0

                Z_k, Z_detail = update_Z(α_k, Z_k, detail=record_iter)
                α_k, α_detail = update_α(α_k, Z_k, detail=record_iter)

                if record_iter:
                    Z_k_star = self._compute_procrustean_transformation(
                        Z_k, Z_hat)
                    result['α'].append(α_k)
                    result['Z'].append(Z_k_star)
                    result['α_detail'].append(α_detail)
                    result['Z_detail'].append(Z_detail)

            acceptance_rates = {
                k: np.mean(list(funcy.pluck(
                    'accepted', result['{}_detail'.format(k)])))
                for k in ['α', 'Z']
            }
            logger.info('Sampling...DONE')
            logger.info('Acceptance rates: {!r}'.format(acceptance_rates))

            return result
        except KeyboardInterrupt:
            logger.info('Sampling...ABORTED')
            logger.info('Acceptance rates: {!r}'.format(acceptance_rates))
            return result


class MonkLatentPositionModel(LatentPositionModel):
    def __init__(self, k, μ_α, σ_α, μ_zi, σ_zi, σ_q_α=0.2, σ_q_zi=0.1):
        super().__init__(k)
        self._pa = scipy.stats.norm(loc=μ_α, scale=σ_α)
        self._pzi = scipy.stats.multivariate_normal(mean=[μ_zi] * k,
                                                    cov=np.diag([σ_zi] * k))
        self.σ_q_α = σ_q_α
        self.σ_q_zi = σ_q_zi

    def logpa(self, α):
        return self._pa.logpdf(α)

    def logpz(self, Z):
        return np.sum(self._pzi.logpdf(Z.T))

    def ηij(self, α, zi, zj):
        '''Compute the log odds for actors i and j in

        The log odds are given as

            logit p(y_{i,j} = 1 | \alpha, z_i, z_j) = \alpha - |z_i - z_j|.

        '''
        dij = zi - zj
        return np.asscalar(α - np.sqrt(dij.dot(dij)))

from numpy import array, log, logaddexp, ones, asarray
from scipy.special import gammaln, gamma, hyp2f1
from scipy.optimize import differential_evolution
from scipy.misc import logsumexp
import numpy as np


# initial a class for pareto/NBD model
class ParetoNBD:
    def __init__(self, pars=None, penalty=0.):
        self.pars = pars
        # penalty is the coefficient to adjust the parameters errors when calculating the LL value
        self.penalty = penalty

    @staticmethod
    def check_inputs(freq, rec, age):
        # check the input data
        # convert the input data to numpy array type
        freq = asarray(freq)
        rec = asarray(rec)
        age = asarray(age)
        # a. rec is always larger than age
        def check_recency_is_larger_than_age(rec, age):
            if any(rec.astype(float) > age.astype(float)):
                raise ValueError('Some customers recency are larger than age. This should be an input error')
        # b. freq should be larger than 0
        def check_frequency_is_zero(freq):
            if(any(freq.astype(int) == 0)):
                raise ValueError('Some customers frequency is zero. This should be an input error and not be imported into model training')
        # c. all input data should be integer
        def check_input_data_is_integer(freq, rec, age):
            if any(freq.astype(float) - freq.astype(int) != 0):
                raise ValueError('Frequency data should not be non-integer')
            if any(rec.astype(float) - rec.astype(int) != 0):
                raise ValueError('Recency data should not be non-integer')
            if any(age.astype(float) - age.astype(int) != 0):
                raise ValueError('Age data should not be non-integer')
        check_frequency_is_zero(freq)
        check_input_data_is_integer(freq, rec, age)
        check_recency_is_larger_than_age(rec, age)
        return freq, rec, age

    @staticmethod
    # A0 is a key part in Maximum likelihood formula which is used to estimate alpha, beta, r, s
    def log_a0(r, alpha, s, beta, freq, rec, age):

        # check if alpha >= beta for format of A0 is slightly from when beta >= alpha
        max_ab, min_ab, b = (alpha, beta, s+1.) if alpha > beta else (beta, alpha, r + freq)
        ai = r + s + freq
        ci = ai + 1.
        zt = (max_ab - min_ab) / (max_ab + rec)
        zT = (max_ab - min_ab) / (max_ab + age)
        # F(a,b,c,z) is Gaussian hypergeometric function
        ft, fT = hyp2f1(ai, b, ci, zt), hyp2f1(ai, b, ci, zT)
        sign = ones(len(freq))
        # result = log(ft * (max_ab + age) ** ai -fT * (max_ab + rec) ** ai) - ai * log((max_ab + rec)*(max_ab + age))
        # logsumexp function is used for calculating the log ratio of A0.
        # why use logsumexp: when computing with very small numbers, it is very common to keep the numbers in logspace.
        # Most of the time, you only want to add up numbers in an array which could be done using log(sum(exp(a)))
        result = logsumexp([log(ft) + ai * log(max_ab + age), log(fT) + ai * log(max_ab + rec)], axis=0, b=[sign, -sign]) - ai * log((max_ab + rec) * (max_ab + age))
        return result

    def pareto_nbd_LL(self, pars, freq, rec, age):
        # log-likelihood function: LL(r, alpha, s, beta) = sigma(ln(L(r,alpha,s,beta|xi,txi,Ti))) + sigma(ln(g(r, alpha, s, beta)))
        # which could be maximized by standard numerical optimization routines
        r, alpha, s, beta = pars
        log_a0_result = self.log_a0(r, alpha, s, beta, freq, rec, age)
        part_fst = gammaln(r + freq) - gammaln(r) + r * log(alpha) + s * log(beta)
        part_snd = logaddexp(-(r + freq) * log(alpha + age) - s * log(beta + age),
                             log(s) + log_a0_result - log(r + s + freq))
        # for we need to maximize the LL, in another word, we need to minimize -LL. (sicpy optimization function is used for minimizing).
        # sum up to calculate the ln(L) and penalize the LL due to the effect of parameters: r, s, alpha, beta
        LL = - (part_fst + part_snd).sum() + self.penalty * log(pars).sum()
        return LL


    def model_pars_fit(self, freq, rec, age):
        # minimize the maximum likelihood function LL
        # we choose scipy.optimize.differential_evolution to find the global minimum of a multivariate function.
        # Differential Evolution is stochastic in nature (does not use gradient methods) to find the minimium, and can search large areas of candidate space.
        # check input data
        freq, rec, age = self.check_inputs(freq, rec, age)
        fit_results = differential_evolution(self.pareto_nbd_LL, bounds=[[1e-3, 20]] * 4, maxiter=400, args=(freq, rec, age), popsize=50, tol=1e-3)
        print fit_results
        self.pars = fit_results.x
        return self


    def p_alive_present(self, freq, rec, age):
        # estimate the possibility that customer is still alive at the present moment
        # check the input data
        freq, rec, age = self.check_inputs(freq, rec, age)
        r, alpha, s, beta = self.pars
        log_a0_result = self.log_a0(r, alpha, s, beta, freq, rec, age)
        p = 1./(1. + (s / (r + freq + s)) * (beta + age) ** s * (alpha + age) ** (r + freq) * np.exp(log_a0_result))
        return p

    def pareto_nbd_Li(self, freq, rec, age):
        # log L
        r, alpha, s, beta = self.pars
        log_a0_result = self.log_a0(r, alpha, s, beta, freq, rec, age)
        part_fst = gammaln(r + freq) - gammaln(r) + r * log(alpha) + s * log(beta)
        part_snd = logaddexp(-(r + freq) * log(alpha + age) - s * log(beta + age),
                             log(s) + log_a0_result - log(r + s + freq))
        return part_fst + part_snd

    def freq_future(self, freq, rec, age, k):
        # estimate the transaction that the customer will make until the future k days (total time period is age + k)
        # check the input data
        freq, rec, age = self.check_inputs(freq, rec, age)
        # check k: it should be an integer and larger than 0
        r, alpha, s, beta = self.pars
        # pareto_pbd_LL return the -log(Li)
        log_Li = self.pareto_nbd_Li(freq, rec, age)
        log_part_a = gammaln(r + freq) + r*log(alpha) + s*log(beta) - gammaln(r) - (r + freq)*log(alpha + age) - s*log(beta + age) - log_Li
        # for s < 1
        log_part_b = log(r + freq) + log(beta + age) - log(alpha + age) - log(1. -s)
        log_part_c = log(np.power((beta + age) / (beta + age + k), s -1.) - 1.)
        freq_future = np.exp(log_part_a + log_part_b + log_part_c)
        return freq_future






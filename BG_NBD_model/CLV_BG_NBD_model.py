__author__ = 'Arkenstone'
from scipy.special import gammaln, betaln, hyp2f1
from scipy.optimize import minimize, differential_evolution
from numpy import log, logaddexp, asarray, power
import numpy as np

class BGNBD:
    def __init__(self, pars=None, penalty=0.):
        self.pars = pars
        self.penalty = penalty

    @staticmethod
    def check_inputs(freq, rec, age):
        # check the input data
        # convert the input data to numpy array type
        freq = asarray(freq)
        rec = asarray(rec)
        age = asarray(age)

        # a. rec is always less than age
        def check_recency_is_larger_than_age(rec, age):
            if any(rec.astype(float) > age.astype(float)):
                raise ValueError('Some customers recency are larger than age. This should be an input error')

        # b. freq should be larger than 0
        def check_frequency_is_zero(freq):
            if any(freq.astype(int) == 0):
                raise ValueError(
                    'Some customers frequency is zero. This should be an input error and not be imported into model training')

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

    def BGNBD_LL(self, pars, freq, rec, age):
        r, alpha, a, b = pars
        a1 = betaln(a, b+freq) -betaln(a, b) + gammaln(r+freq) + r*log(alpha) - gammaln(r) - (r+freq)*log(a+age)
        # for we only analyze customers with more than one freq, so the delta parameter in the second part always equals to 1
        a2 = betaln(a+1, b+freq-1) - betaln(a, b) + gammaln(r+freq) + r*log(alpha) - gammaln(r) - (r+freq)*log(alpha+rec)
        # for the BGNBD_LL function should be maximized to get the optimized parameters while minimize method is used for minimization,
        # so we use negative form of BGNBD_LL function for model training.
        neg_LL = -(a1 + a2).sum() + self.penalty * log(pars).sum()
        return neg_LL

    def fit_BG_NBD_pars(self, freq, rec, age):
        # freq, rec, age should be np.array type, since calculation in the BFNBD_LL model is calculated using np package
        pars = np.array([1., 1., 1., 1.])
        # check freq, rec and age: value and type (should be np.array)
        freq, rec, age = self.check_inputs(freq, rec, age)
        # use minimize method
        # fit_result = minimize(self.BGNBD_LL, x0=pars, method='nelder-mead', tol=1e-4, options={'maxiter': 400})
        # use differential_evolution method
        fit_result = differential_evolution(self.BGNBD_LL, bounds=[[1e-3, 20]] * 4, args=(freq, rec, age), maxiter=400, tol=1e-4, popsize=50)
        print fit_result
        self.pars = fit_result.x
        return self

    def p_alive_present(self, freq, rec, age):
        # get possibility of the customer is still alive after age time (now)
        freq, rec, age = self.check_inputs(freq, rec, age)
        # for freq is always larger than 0
        r, alpha, a, b = self.pars
        p_alive = 1./(1. + a/(b+freq-1)*power((alpha+age)/(alpha+rec), r+freq))
        return p_alive

    def freq_future_k_days(self, freq, rec, age, k):
        # predict transactions will happen in next k days
        r, alpha, a, b = self.pars
        freq, rec, age = self.check_inputs(freq, rec, age)
        a1 = (a+b+freq-1.)/(a-1.)
        a2 = 1. - power((alpha+age)/(alpha+age+k), r+freq) * hyp2f1(r+freq, b+freq, a+b+freq-1, k/(alpha+age+k))
        a3 = 1. + a/(b+freq-1) * power((alpha+age)/(alpha+rec), r+freq)
        future_freq = a1 * a2 / a3
        return future_freq

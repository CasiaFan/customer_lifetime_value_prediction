# Prediction of Customer Lifetime Value with RFM data using Pareto/NBD model
## Reference:
**1. Peter Fader et.al, 2005, A Note on Deriving the Pareto/NBD Model and Related Expressions** <br>
**2. Nicolas Glady et.al, 2009, A modified Pareto/NBD approach for predicting customer lifetime value**

## 1. Background
The customer lifetime value (CLV) is the discounted value of the future profits that this customer yields to the company. Specifically, we need to predict the futrue number of transactions a customer will make and profit of every transaction. Here, Pareto-NBD model could be appplied in this issue and these two key parameters will be estimated separately.

Nomrally, CLV is a function of all transactions a customer will made in the futrue.
![fig1](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/1.png) <br>
d: discount rate (assumed to be constant). CFi,k: net cash flow made by customer i during period k. But here the time dependency factor is not considered, for we study the moment of prediction of the CLV is identical for all customers. Prediction of CLV use a two step scheme.
- First forcast the futrue number of transactions of each individual.
- Then individual average profit per transaction is estimated.

Pareto/NBD model was developed by Schmittlein et al. (1987). This model derive expressions for 2 things:
- the probability a customer is still alive with a given transaction history
- expected number of future transactions for a customer with given transaction history
After estimating the parameters of the Pareto/NBD model, we could forcast the future activity of a customer. <br>
Three past purchasing behavior measures are required for every customer i.  <br>
- Ti: time between the entry of the individual i as a customer of the company until now.
- ti: the time between the entry date and the last purchase day, in another word, the recency. The more recent is the last purchase, the higher ti will be, and 0 < ti < Ti.
- xi: the number of transactions the customer i has made after k time units -- the frequency

## 2. Details of Pareto/NBD model -- methodology
### a. Model assumption
1. If no interference, customers go through two stages in their “lifetime” with a specific firm: they are “alive” for some period of time, then become permanently inactive.
2. While alive, the number of transactions made by a customer follows a Poisson process with transaction rate λi; If the customer i is still active at Ti (so τi > Ti), the number of
purchases xi in (0; Ti] has the Poisson distribution
![fig2](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/2.png) <br>
3. Each customer remains active during a time being exponentially distributed with death rate μi
![fig3](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/3.png) <br>
Since the parameters λi and μi can be different among customers, the Pareto/NBD model makes 2 assumptions on the heterogeneity across customers.
4. Heterogeneity in transaction rates λi across customers follows a gamma distribution with shape
parameter r and scale parameter α: (with E[λi|r,α] = r/α)
![fig4](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/4.png) <br>
5. Heterogeneity in dropout rates μi across customers follows a gamma distribution with shape
parameter s and scale parameter β: (with E[μi|s,β] = s/β)
![fig5](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/5.png) <br>
6. the purchasing rates λi and the death rate μi are considered as distributed independently of each other.
The parameters r, α, s and β are unknown and need to be estimated.

### b. Estimate r, α, s and β using Maximum Likelihood (MLE)
The likelihood for an individual i with purchase history (xi, ti, Ti) is here, removing the conditioning on the λi and μi: <br>
![fig6a](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/6a.png) <br>
After several transformation, Li equals to: <br>
![fig6](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/6.png) <br>
as for A0, if α >= β: <br>
![fig7](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/7.png) <br>
if α < β:
![fig8](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/8.png) <br>
F is the Gaussian hypergeometric function: <br>
![fig9](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/9.png) <br>
where (a)j is Pochhammer’s symbol which denotes the ascending factorial a(a+1) · · · (a+j −1).

Suppose we have a sample of N customers, where customer i had xi transactions in the period (0, Ti], with the last transaction occurring at ti. The sample log-likelihood function is: <br>
![fig10](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/10.png) <br>
This could be maximized by standard numerical optimization routines, then parameters r, α, s, β are estimated. <br>
**Note: here we use [differential_evolution](http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.differential_evolution.html) function of scipy.optimize package. Since it finds the global minimum of a multivariate function, the input function should be -LL(r, α, s, β)**

### c. Derivation of P_alive at time Ti
The probability that a customer with purchase history (xi, ti, Ti) is “alive” at time Ti is the probability that the (unobserved) time at which he becomes inactive (τ) occurs after T -- P(τ > T).<br>
![fig11](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/11.png) <br>
where A0 is mentioned previously.

### d. Derivation of conditional expectation
The expected number of purchases in the period (T, T +t] for a customer with purchase history (xi, ti, Ti), which we call conditional expectation, E(Y(t)|xi,ti,Ti). <br>
![fig12](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/12.png) <br>
where where L is the likelihood of equation with estimated parameters, and Γ(.) denotes the standard Gamma function.

### input data
- a. from csv file with customers' information required above: frequency xi, duration Ti, recency ti
- b. from mysql database with the same information above
**Tips: in order to test prediction precision, we could choose first k days as historical purchase data while the rest are furture T-k days for testing**

### scritps
The program is writen in python mainly with scipy package, eg: scipy.special, scipy.optimize, scipy.misc. The key function is differential_evolution which is mentioned in part b.

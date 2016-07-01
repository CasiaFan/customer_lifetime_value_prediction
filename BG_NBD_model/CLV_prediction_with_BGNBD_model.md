# Prediction of Customer Lifetime Value with RFM data using BG/NBD model
## Reference:
**1. Peter Fader et.al, 2005, “Counting Your Customers” the Easy Way: An Alternative to the Pareto/NBD Model** <br>
**2. Peter Fader et.al, 2008, Computing P(alive) Using the BG/NBD Model**

## 1. Background
The customer lifetime value (CLV) is the discounted value of the future profits that this customer yields to the company. Specifically, we need to predict the futrue number of transactions a customer will make and profit of every transaction. Here, BG/NBD model could be appplied in this issue and these two key parameters will be estimated separately.

Nomrally, CLV is a function of all transactions a customer will made in the futrue.
![fig1](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/1.png) <br>
d: discount rate (assumed to be constant). CFi,k: net cash flow made by customer i during period k. But here the time dependency factor is not considered, for we study the moment of prediction of the CLV is identical for all customers. Prediction of CLV use a two step scheme.
- First forcast the futrue number of transactions of each individual.
- Then individual average profit per transaction is estimated.

BG/NBD model is similar to Pareto/NBD model. The only difference is about how/when customers becomes inactive. Pareto/NBD model assumes dropout can occur at any point in time, while the BG/NBD model assumes instead that the dropout will occure immediately after a purchase. This model derive expressions for 2 things:
- the probability a customer is still alive with a given transaction history
- expected number of future transactions for a customer with given transaction history
After estimating the parameters of the BG/NBD model, we could forcast the future activity of a customer. <br>
Three past purchasing behavior measures are required for every customer i.  <br>
- Ti: time between the entry of the individual i as a customer of the company until now.
- ti: the time between the entry date and the last purchase day, in another word, the recency. The more recent is the last purchase, the higher ti will be, and 0 < ti < Ti.
- xi: the number of transactions the customer i has made after k time units -- the frequency

## 2. Details of BG/NBD model -- methodology
### a. Model assumption
1. If no interference, customers go through two stages in their “lifetime” with a specific firm: they are “alive” for some period of time, then become permanently inactive.
2. While alive, the number of transactions made by a customer follows a Poisson process with transaction rate λi; If the customer i is still active at Ti (so τi > Ti), the number of purchases xi in (0; Ti] has the Poisson distribution
![fig2](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/BG_NBD_model/pic/5.png) <br>
3. After any transaction, a customer becomes inactive with probability p. The point at which the customer “drops out” is distributed across transactions according to a (shifted) geometric distribution with pmf
![fig3](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/BG_NBD_model/pic/7.png) <br>
Since the parameters λi and p can be different among customers, the BG/NBD model makes 2 assumptions on the heterogeneity across customers.
4. Heterogeneity in transaction rates λi across customers follows a gamma distribution with shape parameter r and scale parameter α: (with E[λi|r,α] = r/α) <br>
![fig4](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/BG_NBD_model/pic/6.png) <br>
5. Heterogeneity in p follows a beta distribution with pdf <br>
![fig5](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/BG_NBD_model/pic/8.png) <br>
B(a,b) is the beta function, which can be expressed in terms of gamma functions: B(a,b)=gamma(a)*gamma(b)/gamma(a+b).
6. the purchasing rates λi and the dropout possibility p are considered as distributed independently of each other.
The parameters r, α, a and b are unknown and need to be estimated.

### b. Estimate r, α, a and b using Maximum Likelihood (MLE)
The likelihood for an individual i with purchase history (xi, ti, Ti) is here, removing the conditioning on the λi and p: <br>
![fig6](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/BG_NBD_model/pic/1.png) <br>

Suppose we have a sample of N customers, where customer i had xi transactions in the period (0, Ti], with the last transaction occurring at ti. The sample log-likelihood function is: <br>
![fig7](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/BG_NBD_model/pic/2.png) <br>
This could be maximized by standard numerical optimization routines, then parameters r, α, s, β are estimated. <br>
**Note: here we use [differential_evolution](http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.differential_evolution.html) function of scipy.optimize package. Since it finds the global minimum of a multivariate function, the input function should be -LL(r, α, s, β)**

### c. Derivation of P_alive at time Ti
The probability that a customer with purchase history (xi, ti, Ti) is “alive” at time Ti is the probability that the (unobserved) time at which he becomes inactive (τ) occurs after T -- P(τ > T).<br>
![fig8](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/BG_NBD_model/pic/3.png) <br>


### d. Derivation of conditional expectation
The expected number of purchases in the period (T, T +t] for a customer with purchase history (xi, ti, Ti), which we call conditional expectation, E(Y(t)|xi,ti,Ti). <br>
![fig9](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/BG_NBD_model/pic/4.png) <br>
where F is Gaussian hypergeometric function.

### Input data
- a. from csv file with customers' information required above: frequency xi, duration Ti, recency ti
- b. from mysql database with the same information above
**Tips: in order to test prediction precision, we could choose first k days as historical purchase data while the rest are furture T-k days for testing**

### Scritps
The program is writen in python mainly with scipy package, eg: scipy.special, scipy.optimize, scipy.misc. The key function is differential_evolution which is mentioned in part b.

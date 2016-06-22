# 基于顾客RFM历史消费数据利用Pareto/NBD模型推测用户生命周期价值（CLV）
## 参考文献:
**1. Peter Fader et.al, 2005, A Note on Deriving the Pareto/NBD Model and Related Expressions** <br>
**2. Nicolas Glady et.al, 2009, A modified Pareto/NBD approach for predicting customer lifetime value**

## 1. 背景
CLV是指该顾客在未来一段时间内能够对公司产生的利润值。具体来说，我们需要通过预测该顾客在未来一段时间的消费次数和每次消费使公司获得的利润从而推算CLV。此处我们将使用Pareto-NBD模型来分别对着两个参数进行估计。

CLV是与顾客消费次数密切相关的函数。如下： <br>
![fig1](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/1.png) <br>
d: 打折率（假设该值不随时间变化）。 CFi,k: 顾客i在k时期内产生的现金流。这里不考虑时间依赖因素，因为我们预测CLV的时间节点对所有用户都一样。 预测CLV需要以下两步：
- 首先预测个人未来消费次数
- 其次计算每次消费能产生的平均收益

Pareto/NBD模型首先由Schmittlein等人于1987年首次提出。该模型演算出以下两者的表达式：
- 有一段历史消费记录的顾客在某时间节点还是活跃状态的概率
- 未来一段时间的内某顾客的消费次数
在估算出Pareto/NBD模型的参数以后，我们就可以预测顾客未来的活动情况。<br>
在此模型中，以下三个顾客消费参数为模型训练所必须的：<br>
- Ti: 顾客i第一次消费到现在的时间长度
- ti: 第一次消费距离上次消费的时间长度( 0 < ti < Ti)
- xi: 顾客i到现在为止的消费次数

## 2. Pareto/NBD模型方法学部分
### a. 模型假设
1. 首先我们假设在没有外界干扰的情况下，顾客的消费行为经历两段时期：开始一段时期的活跃，之后不再有任何购买行为。
2. 当顾客还是活跃的时候，单个顾客的购买次数符合参数为λi的泊松过程。如果顾客到Ti时期还是活跃的， 那再（0，Ti]这段时间内的购买次数符合泊松分布 <br>
![fig2](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/2.png) <br>
3. 每个顾客在一段时间之后仍然保持活跃符合流失率为μi的指数分布。<br>
![fig3](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/3.png) <br>
由参数λi和μi在每个顾客上都可能是不同的，因此模型还假设了两条关于顾客异质性的假设。
4. 不同顾客间的交易率λi异质性符合形状参数为s，尺度参数为α的gamma分布：(E[λi|r,α] = r/α) <br>
![fig4](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/4.png) <br>
5. 不同顾客间流失率μi的异质性符合形状参数为s，尺度参数为β的gamma分布：(E[μi|s,β] = s/β) <br>
![fig5](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/5.png) <br>
6. 购买率λi和流失率μi的分布彼此独立。因此，参数r, α, s， β是未知的，需要计算。

### b. 利用极大似然（MLE）估计参数r, α, s， β
历史消费记录为(xi, ti, Ti)的单个顾客i的似然如下所示, 消掉的参数λi和μi: <br>
![fig6a](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/6a.png) <br>
经过一系列的变换，Li可化为: <br>
![fig6](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/6.png) <br>
对于A0, 如果α >= β: <br>
![fig7](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/7.png) <br>
如果α < β: <br>
![fig8](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/8.png) <br>
F是高斯超几何分布函数（Gaussian hypergeometric）: <br>
![fig9](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/9.png) <br>
(a)j：Pochhammer’s symbol，为下降阶乘幂a(a+1) · · · (a+j −1).

如果我们有N个顾客的样本，顾客i在(0, Ti]期间有xi次消费记录，最后一次消费发生在ti。改样本的log-likelihood函数是: <br>
![fig10](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/10.png) <br>
该函数可以通过标准数值优化路径实现**最大化**，此时得到最优的参数r, α, s, β。 <br>
**注意： 此处我们使用scipy.optimize包的 [differential_evolution](http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.differential_evolution.html) 函数。由于此函数最后是将函数值最小化，所以在输入模型的时候的函数应该为上式取反 -LL(r, α, s, β)**

### c. 计算Ti时间顾客仍然活跃的概率P_alive
历史消费数据为(xi, ti, Ti)的顾客i在Ti仍然活跃的概率就是在Ti之后顾客不再活跃的概率 P(τ > T)。 <br>
![fig11](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/11.png) <br>
A0如之前所示。

### d.计算顾客消费次数的条件期望
在(T, T +t]时期内顾客i的期望消费次数我们称为条件期望： E(Y(t)|xi,ti,Ti)。 <br>
![fig12](https://github.com/CasiaFan/customer_lifetime_value_prediction/blob/master/pic/12.png) <br>
L如之前所示为似然的函数，而Γ(.)则是标准Gamma函数.

### 输入数据
此处我们输入的recency数据时现在距离上次消费的时间，与前文定义的相反，所以在输入模型计算之前需要进行Ti-ti转换成所需的变量。<br>
这里输入模型的数据可以来自文本或数据库，本次使用的是数据库。<br>
**提示: 如果想要测试模型输出的准确性，可以将历史数据拆分成两份，前半部分作为模型训练，后半部分作为检验。**

### 脚本
本次脚本使用python编写，主要利用了scipy包的一些函数，例如scipy.special, scipy.optimize, scipy.misc。其中，关键的函数为 differential_evolution， 已经在b部分出有提到，具体参数设置可以点开链接查看。.

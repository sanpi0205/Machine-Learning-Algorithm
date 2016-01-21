# 决策树算法



## 模型



决策树是利用信息理论中熵 (entropy) 或信息增益 (information gain) 来度量数据混杂程度，在划分数据时

选择使熵或者信息增益最大的变量（特征）将数据划分为不同分支。



决策树首先要计算数据的熵，熵的计算公式如下



$$

H = - \sum_{i=1}^{n} p(x_i) log_2 p(x_i) 

$$



## 主要算法

决策树的算法有很多种，如ID3、CART、C4.5等等



## Classification and Regression Tree (CART)

CART算法是决策树算法中非常级经典的一种算法，通过对数据空间的某种划分来预测Y或者对Y分类，

假设有$n$个样本如$(x_i, y_i) i=1,2,...,n$，决策树算法需要找到划分数据的变量以及该变量的划分点。

假设存在一个划分，将数据划分为$M$个区域，$R_1,R_2,..,R_M$，决策树回归的模型形式为：



$$

f(x) = \sum_{m=1}^{M} c_m I( x \in R_m)

$$



如果定义模型的损失函数为$ \sum (y_i - f(x_i))^2$，那么很容易推导出最佳的 $\hat{c}_m$ 就是该

区域中$y_i$的均值：



$$

\hat{c}_m = avg( y _i | x_i \in R_m)

$$



因而算法的目的就是要找到最佳的划分 (partition) 最小化均方误差 (sum of squares)。对于任意一个

划分变量 j 和 划分点 s，其对数据的划分可以表示为：



$$

R_1(j, s) = \{X | X_j \leq s \} \quad and \quad R_2(j, s) = \{X  | X >  s  \}

$$



现在要选择最佳的 j 和 s 最小化目标函数：



$$

\underset{j,s}{min} [  \underset{c_1}{min} \sum_{x_i \in R_1(j,s)} (y_i - c_1)^2  \quad + \quad \underset{c_2}{min} \sum_{x_i \in R_2(j,s)} (y_i - c_2)^2 ]

$$



通过扫描数据可以求出最佳的 j 与 s， 通过递归算法我们可以得到将数据划划分细致到每个树节点，但很

明显这样很容易使数据产生过拟合的问题，因而决策树的规模就成为影响该模型性能的最重要因素。



通常的策略是在决策树递归过程中设置一个停止条件，比如当最小节点规模为5的时候 (node size)，停止递归。此时得到一个比较大的树$T_0$，然后通过条件对树进行裁剪。假设$T \subset T_0$








##参考文献：

[1] [wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)

[2] [The Elements of Statistics Learning]()

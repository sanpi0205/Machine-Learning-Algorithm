# 决策树算法



## 模型



决策树是利用信息理论中熵 (entropy) 或信息增益 (information gain) 来度量数据混杂程度，在划分数据时

选择使熵或者信息增益最大的变量（特征）将数据划分为不同分支。



决策树首先要计算数据的熵，熵的计算公式如下



$$

H = - \sum_{i=1}^{n} p(x_i) log_2 p(x_i) 

$$



## 算法

决策树的算法有很多种，如ID3、CART、C4.5等等

参考文献：
[1] (wikipedia)[https://en.wikipedia.org/wiki/Decision_tree_learning]
[2] (sfd)
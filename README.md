# machine-learning-algorithm-library
项目主体方案如下：  
1、构建流程工业基于大数据的人工智能算法库  
流程工业基于大数据的人工智能算法库主要包括以下几类基本算法：（1）变量选择算法；（2）特征提取算法；（3）空间投影算法；（4）聚类分析算法；（5）智能分类算法；（6）智能优化算法。通过调用算法库算法可实现过程数据的离线分析，也可实现基于数据的过程在线运行状态监测。具体包括以下几项功能：（1）过程数据分析与特征挖掘；（2）关键性能指标智能建模；（3）运行状态智能监测与诊断；（4）过程运行状态智能优化。算法库构成及功能如图1所示，具体内容如下。  
![image](https://github.com/xfysq1/machine-learning-algorithm-library/blob/main/%E8%BD%AF%E4%BB%B6%E6%9E%B6%E6%9E%84.png)  
图1基于大数据的人工智能算法库构成及功能  
（1）变量选择算法。变量选择对于过程关键性能指标影响因素分析、预测与监测都发挥着重要作用。通过变量选择算法，可以对影响过程关键性能指标的变量进行选择与分析，提升预测模型鲁棒性与状态监测可靠性。  
1）线性相关分析  
2）互信息分析  
3）最大相关最小冗余分析  
4）回归和独立性的变量重要性    
5）稀疏线性回归变量分析（LASSO）  
6）稀疏神经网络  
7）XGBoost  
（2）特征提取算法。过程数据通常具有高维、复杂相关且冗余的特点。通过特征提取可挖掘过程主要信息，消除冗余信息，实现过程数据信息的简洁表征。  
1）主成分分析方法  
2）偏最小二乘分析方法  
3）核主成分分析方法  
4）流形学习方法（LLE、t-SNE等）  
5）深度神经网络方法（栈式自编码器）  
（3）空间投影方法。数据在高维空间的拓扑结构难以直观显示或呈现。空间投影将高维数据投影到低维空间，可直观表示数据在高维空间的拓扑结构。  
1）线性判别分析  
2）拓扑保留映射（自组织映射神经网络）  
3）深度神经网络  
4）流形学习方法（t-SNE）  
（4）聚类分析算法。历史数据集中样本类标签通常不明确、不完整。通过聚类分析算法可将具有相同或相似拓扑结构、或者距离相近的样本聚成一类，再对每类数据进行分析。  
1）K-均值聚类  
2）高斯混合模型聚类（GMM）  
（5）智能分类算法。智能分类算法采用历史标签数据训练分类模型，通过模型将在线数据归到最相似或关系最紧密的类中。  
1）线性分类器（LDA）  
2)支持向量机（SVDD）  
3)决策树  
4）随机森林  
5)人工神经网络分类器  
6）贝叶斯分类器  
（6）智能优化算法。流程工业过程呈高维、高度非线性特征，且不同过程复杂程度各不相同，求解最优操作条件、模型参数等困难。智能优化算法是这类问题，对操作条件、模型结构及参数寻优的有力求解工具。  
1）粒子群算法  
2）遗传算法  
3）差分进化算法  


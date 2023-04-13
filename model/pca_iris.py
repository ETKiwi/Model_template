# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:20:41 2023

@author: Kiwi
"""

from sklearn.decomposition import PCA
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


# 載入資料集 : 鳶尾花
iris = datasets.load_iris()
x = iris.data

# PCA對特徵降維
pca=PCA(n_components=2) # n_components是要留下的維度，也可以給'mle'，通過最大概似法決定合適的components數量
x_pca = pca.fit(x).transform(x) # 找到特徵映射後資料變異量最大的投影向量只要fit X 即可

# 檢視降維結果
# 降維後的特徵數量，即主成分的數量 (經過映射後的新特徵不是原本特徵)
pca.n_components_ 
# 主成分的方差，即降維後每個主成分所解釋的變異量
pca.explained_variance_
# 新特徵的解釋能力，即每個主成分所解釋的變異量占總變異量的比例
pca.explained_variance_ratio_
# 主成分負荷向量，即每個主成分在原特徵空間中的投影方向
pca.components_ 
# 把解釋能力累加起來 (特徵貢獻度)，可以用來評估主成分的重要性，通常用於選擇合適的主成分數量
np.cumsum(pca.explained_variance_ratio_)

# 將 PCA 結果進行視覺化
plt.scatter(x_pca[:, 0], x_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()



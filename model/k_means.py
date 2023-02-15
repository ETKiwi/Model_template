# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:19:32 2023

@author: BK
"""
from sklearn.datasets import make_blobs # 聚類資料生成
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 檢視資料_散佈圖
def plt_scatter(x, y, title = 'scatter'):
    plt_chinese
    
    plt.scatter(
        x[:,0], x[:,1], # 資料
        c = y,# 顏色_指定
        edgecolor = 'none', # 邊框
        alpha = 0.5 # 透明度
        )
    # 圖示說明    
    plt.title(title) # 標題
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

def plt_chinese():
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 顯示中文
    # plt.rcParams['axes.unicode_minus'] = False # 顯示負號
    
# 建立隨機資料
def build_data():
    # 隨機產生 5 組 2 features 的資料 500 筆
    x, y = make_blobs(
        n_samples = 500, # 樣本數
        n_features = 2, # 每個樣本的特徵數
        centers = 5, # 類別數
        cluster_std = 0.5 # 類別的標準差
    )
    
    return x, y

# 建立模型
def k_means(x):
    # KMeans 分群
    clf = KMeans(n_clusters = 2)   # 設定為2群
    clf.fit(x) # 訓練
    labels = clf.labels_ # 分類內容
    centers = clf.cluster_centers_ # 群心
    y_pred = clf.predict(x)      # 分群

    return x, labels, centers, y_pred

# 建立隨機資料
x, y = build_data()
# 檢視生成的資料
plt_scatter(x, y, '生成的資料')
# 執行模型
x, labels, centers, y_pred = k_means(x)
# 檢視模型分類結果
plt_scatter(x, y_pred, 'k_means分類結果')


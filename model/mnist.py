# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:39:26 2023

@author: Kiwi
"""
'''一個簡單的手寫數字辨識的神經網路模型
'''
# 導入TensorFlow框架
import tensorflow as tf

# 載入 MNIST 數字 dataset
mnist = tf.keras.datasets.mnist
# 將數據集分為訓練集和測試集，x_train和x_test為數字圖像的像素值，y為數字圖像對應的標籤
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# 將數字圖像像素值進行歸一化處理
x_train, x_test = x_train /  255.0, x_test / 255.0

# 建立神經網路，包含四層Flatten, Dense, Dropout, Dense
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # 將二維的圖像數據攤平為一維的數據
  tf.keras.layers.Dense(128, activation='relu'), # 全連接層，128個神經元，使用 ReLU 激活函數
  tf.keras.layers.Dropout(0.2), # 丟棄部分神經元，防止過擬合
  tf.keras.layers.Dense(10, activation='softmax') # 全連接層，10個神經元，使用 softmax 激活函數，得到10個類別的概率分布
])

# 設置模型的損失函數、優化器optimizer和評估指標
model.compile(optimizer='adam', # 使用 Adam 自適應優化器，自動調節學習率
              loss='sparse_categorical_crossentropy', # 使用交叉熵作為損失函數
              metrics=['accuracy']) # 設置評估指標
# 訓練模型
model.fit(x_train, y_train, epochs=5) # 使用訓練集訓練模型，訓練5個epochs

# 對模型進行測試，計算模型在測試集上的準確率 # loss # accuracy
model.evaluate(x_test, y_test)


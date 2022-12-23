# Model_template
This is Machine learning template for python with sklearn

## requirements.txt 
- package in 'requirements.txt', version as below:
```
numpy == 1.20.3
pandas == 1.3.4
matplotlib == 3.4.3
torch == 1.10.2
torchvision == 0.11.3
tensorflow == 2.8.0
scikit-learn == 0.24.2
```
*執行 pip install -r requirements.txt 進行安裝。

## my_temp
```
|_ main.py
   utils.py
   pipeline
   |_ pipeline.py
      steps
      |_ model.py
         preglight.py
         preprocess.py
         step.py
 ```

## main.py
 執行

## utils.py
 延伸工具, 包含繪圖,....

## step.py
 程式架構(abstract method)

## preflight.py
 預設為初步initial使用

## preprocess.py
 資料前處理內容,
 包含資料轉換, na清洗,...

## model.py
 模型建製內容, 
 包含模型, imbalanced data
 
## 備註



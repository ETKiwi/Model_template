# Model_template
This is Machine learning template for python with sklearn

## requirements.txt 
- 模組套件整理在requirements.txt，版本如下:
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

## step.py
 程式架構(abstract method)

## preflight.py
 空的內容，方便之後擴展使用。

## preprocess.py
 資料前處理內容，包含資料轉換, na清洗,...

## model.py
 模型建製內容，包含模型, imbalanced data


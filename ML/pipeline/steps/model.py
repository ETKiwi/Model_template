# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:16:47 2022

@author: BK
"""
import pandas as pd
import pickle
# import re
from sklearn.utils.validation import column_or_1d
# from sklearn import preprocessing
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
# from sklearn.impute import KNNImputer # 補遺
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from imblearn.over_sampling import SMOTE


class Model_:
    
    def process(self, data, inputs, utils):
        '''main
        '''
        path, x, y = inputs['path'], inputs['x'], inputs['y']
        # predict y with clf
        metric_clf_, metric_smote = self.model_clf(
            data, #data_after_predict, 
            path, x, y,
            RandomForestClassifier())
        # predict y with regr
        metric_regr_ = self.model_regr(
            data, #data_after_predict, 
            path, x, y,
            RandomForestRegressor())
        
        to_excel = {
            'data': data, #data_after_predict,
            # 'data append':dt_append,
            'metric_clf':metric_clf_,
            'metric_with_smote':metric_smote,
            'metric_regr':metric_regr_,
                 }
        
        self.data_to_excel(to_excel, self.path, '評估表.xlsx')
        
        return metric_clf_, metric_smote, metric_regr_
    
    def SMOTE_(self, x_train, y_train):
        # SMOTE
        try:
            x_res, y_res = SMOTE(random_state = 6).fit_resample(x_train, y_train)
        except ValueError as e:
            # label_encoder = preprocessing.LabelEncoder()
            # y_train = label_encoder.fit_transform(y_train)
            print(e)
        
        return x_res, y_res
    
    def split(self, test_size = 0.2, random_state = 6):
        data = self.data
        
        data = data[self.x + self.y].dropna()
        X = data[self.x]#.drop(columns = x)
        Y = column_or_1d(data[self.y])
        x_train, x_test, y_train, y_test =\
            train_test_split(X, Y, test_size = 0.2, random_state = 6)
        
        return x_train, x_test, y_train, y_test
        
    def model_clf(self, model):
        '''spilit data for model
        '''
        data = self.data
        Pr = Preprocess()
        
        x_train, x_test, y_train, y_test = self.split(data)
        # imbalanced data
        x_res, y_res = Pr.SMOTE_(x_train, y_train)
    
        # model classifier
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metric_ = self.metric_clf( # feature_importances,
            data, model, x_train, y_train, x_test, y_test, y_pred)
        
        # model classifier with imbalanced data
        model_smote = model.fit(x_res, y_res)
        y_pred_smote = model_smote.predict(x_test)
        metric_smote = self.metric_clf( # feature_importances, 
            data, model, x_train, y_train, x_test, y_test, y_pred_smote)
        # save model
        self.model_to_pickle(model)
        
        return metric_, metric_smote # feature_importances,
    
    def model_regr(self, model):
        '''spilit data for model
        '''
        data = self.data
        
        x_train, x_test, y_train, y_test = self.split(data)
    
        # build model
        # model = RandomForestRegressor()
        
        # tension model-regr
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        # 
        metric_ = self.metric_regr(
            data, model, x_train, y_train, x_test, y_test, y_pred)
        # save model
        self.model_to_pickle(model)
    
        return metric_ # feature_importances,
    
    def metric_clf(self, model, x_train, y_train, x_test, y_test, y_pred):
        '''metric for classifier
        '''
        data = self.data
        # metric score data
        metric_ = pd.DataFrame([{
            '訓練集_預測成功': model.score(x_train,y_train),# train_score
             '測試集_預測成功(accuracy)':model.score(x_test,y_test),#test_score
             'accuracy': accuracy_score(y_test, y_pred),
             '預測正確數': accuracy_score(y_test, y_pred, normalize=False),# correct_data
             'Precision': precision_score(y_test, y_pred)*100,
             'Recall': recall_score(y_test, y_pred)*100,
             'F1': f1_score(y_test, y_pred)*100,
             'Confusion_Matrix': confusion_matrix(y_test, y_pred),
             '總數': len(data)
             }])
    
        return  metric_#feature_importances,
    
    def feature(self, model):
        # feature importance
        feature_importances = pd.DataFrame(
            model.feature_importances_, index = self.x).T
    
        return feature_importances
    
    def metric_regr(self, model, x_train, y_train, x_test, y_test, y_pred):
        '''metric for regression
        '''
        data = self.data
        # metric score data
        metric_ = pd.DataFrame([{
                "r2": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred), 
                "RMSE": mean_squared_error(y_test, y_pred, squared=True),
                '參數':data.columns,
                '總數': len(data)
             }])
    
        return metric_#feature_importances,

    
    def model_to_pickle(self, model):
        # data = self.data
        model_pkl = self.path + '\\pickle\\stair_model.pkl'
        with open(model_pkl, 'wb') as f:
            pickle.dump(model, f)
    
    def data_to_excel(self, excel_name):
        data = self.data
        with pd.ExcelWriter(self.path + excel_name) as writer:
            for excel_name, dt in data.items():
                dt.to_excel(writer, excel_name, index = False)
    









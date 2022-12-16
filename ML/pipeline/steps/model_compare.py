# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:42:21 2022

@author: BK
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import xgboost as xgb
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


class ModelCompare:
    def clf(self, x_train, y_train):
        # Algorithms
        # models = []
        # models.append(('LDA', LinearDiscriminantAnalysis()))
        # models.append(('KNN', KNeighborsClassifier()))
        # models.append(('CART', DecisionTreeClassifier()))
        # models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=1)))
        # models.append(('SVM', SVC(gamma='auto')))
        models = {
            "RF" : RandomForestClassifier(),
            "DT" : DecisionTreeClassifier(),
            "LR" : LogisticRegression(),
            "GB" : GradientBoostingClassifier(),
            "SVC" : SVC(),
            "KNN":  KNeighborsClassifier(),
            "AdaB" : AdaBoostClassifier(),}
        
        self.model_compare(models, x_train, y_train, scoring_='accuracy')
    
    def regr(self, x_train, y_train):
        # model regression
        models = {
            "RF" : RandomForestRegressor(),
            "XGB" : xgb.XGBRegressor(),
            "Lin" : LinearRegression(), 
            "GB" : GradientBoostingRegressor(),
            "KNN" : KNeighborsRegressor(), 
            "DT" : DecisionTreeRegressor(),
            "SVR" : SVR(),
            "AdaB" : AdaBoostRegressor()
            }
        
        self.model_compare(models, x_train, y_train, scoring_='r2')
    
    
    def model_compare(self, models, x_train, y_train, scoring_):
        # evaluate each model in turn
        results = []
        names = []
        
        # for name, model in models:
        for name in models:
        	# kfold = StratifiedKFold(n_splits=10, random_state=6, shuffle=True)
            #使用 cv 及 validation data 計算  error 評估模型
            model = models[name]
            
            cv_results = cross_val_score(model, x_train, y_train,
                cv = KFold(n_splits = 10), # kfold, 
                scoring = scoring_)
            
            results.append(cv_results)
            names.append(name)
            print(f'{name}, metic_mean : {cv_results.mean()}, metic_std : {cv_results.std()}')
        
        # model_compare_plot(results, names)
        
    
    
    def model_compare_plot(self, results, names):
        # Compare Algorithms
        plt.boxplot(results, labels=names)
        plt.title('Algorithm Comparison')
        plt.show()
    
    def metric(self, y_test, predictions):
        # Evaluate predictions
        # # 0:'setosa', 1:'versicolor', 2:'virginica'
        print(accuracy_score(y_test, predictions)) #準確率
        print(confusion_matrix(y_test, predictions)) # 混淆矩陣confusion matrix
        print(classification_report(y_test, predictions))




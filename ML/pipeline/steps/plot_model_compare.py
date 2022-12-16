# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:43:41 2022

@author: user
"""
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc


class PlotModel:
    '''Regression Model Plot
    '''
    def __init__(self):
        '''Plot Chinese Problem'''
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] #runtime configuration parameters
        plt.rcParams['axes.unicode_minus'] = False

    def pred_plot(self, path, Y_test, predict, title_name):
        '''Y_test vs Predict Result'''
        x_axis = np.array(range(0, predict.shape[0]))
        
        plt.plot(x_axis, predict, linestyle="--", marker=".", color='r', label="pred")
        plt.plot(x_axis, Y_test, linestyle="--", marker=".", color='g', label="Y_test")
        plt.legend(loc ='upper left')
        
        self.plot_(title_name, "test_number", "A.U.", path)
        # plt.xlabel('test_number')
        # plt.ylabel("A.U.")
        # plt.savefig(path)
        # plt.show() 

    def residual_plot(self, path, Y_test, predict, title_name):
        '''show each cell residual(predict-test) after predict'''
        residual = predict.reshape(-1) - Y_test
        sns.residplot(Y_test, residual, lowess=True, color="g")
        
        self.plot_(title_name, "Y", "Residual", path)
        # plt.xlabel('Y')
        # plt.ylabel("Residual")
        # plt.title(title_name)
        # plt.savefig(path)  
        # plt.show()

    def dist_(self, path, Y_test, predict, title_name):
        '''dist: kde plot + histogram plot'''
        residual = predict.reshape(-1) - Y_test
        sns.distplot(residual)
        
        self.plot_(title_name, "Value", "Number", path)
        # plt.title(title_name)
        # plt.xlabel("Value")
        # plt.ylabel("Number")
        # plt.savefig(path)
        # plt.show()

    def hist_(self, path, Y_test, predict, title_name):
        '''histogram: predictions & Ytest'''
        plt.hist(predict, edgecolor = 'white')
        plt.grid(True)
        
        self.plot_(title_name, "Value", "Number", path)

    def boxplot_(self, model_results, model_names,
                 path, plot_name, type_name = 'regr'):
        figure = plt.figure()
        figure.suptitle(f'{plot_name}_{type_name}_model_compare', fontsize=12)
        axis = figure.add_subplot(111)
        plt.boxplot(model_results)
        axis.set_xticklabels(model_names, rotation = 45, ha="right", fontsize=12)
        axis.set_ylabel('r2', fontsize = 12)
        plt.margins(0.1, 0.1)#Plot edge
        plt.savefig(path)
        plt.show()
    
    # def roc_(self,path, title, fper, tper):
    #     roc_auc = auc(fper, tper)
    #     # plot
    #     plt.plot(fper, tper, color='red', label= 'AUC = %0.2f' % roc_auc)
    #     plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(title) # Receiver Operating Characteristic Curve
    #     plt.legend(loc = 'lower right')
    #     plt.savefig(path)
    #     plt.show()
        
    def roc_curve_(self,path,y_test, y_pred, title_ = ''):
        # tpr: true positive, fpr: false negative rate
        fpr, tpr, thersholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        
        title_ = 'ROC' + title_
        plt.figure(dpi = 200)
        plt.title(title_)
        plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'best')
        plt.plot([0, 1], [0, 1],'red','--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(path)
        plt.show()
        
    def plot_(self, title, xlabel_, ylabel_, path):
        plt.title(title)
        plt.xlabel(xlabel_)
        plt.ylabel(ylabel_)
        plt.savefig(path)
        plt.show()

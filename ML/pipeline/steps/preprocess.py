# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:17:21 2022

@author: user
"""
import pandas as pd

from ML.pipeline.steps.step import Step

class Preprocess(Step):
    
    def process(self, data, inputs, utils):
        dt_append = pd.DataFrame()
        for file in inputs['files']:
            data = pd.read_csv(
                inputs['path'] + file, encoding='Big5', engine='python',
                usecols = inputs['x'])
            # duplicates
            # for column in inputs['column_duplicate']:
                # data = self.drop_duplicates(data, column).reset_index(drop = True)
            # useless data
            data = self.drop_useless(data).reset_index(drop = True)
            
            # encoder
            for column in inputs['column_encoder']:
                data = self.process_trans(data, column).reset_index(drop = True)
                
            # outlier
            for column in inputs['x']:
                data = self.process_outlier(data, column).reset_index(drop = True)

            dt_append = dt_append.append(data)
        dt_append = dt_append.reset_index(drop = True)
        
        return dt_append
    
    
    def drop_duplicates(self, data, column):
        '''drop duplicates data
        '''
        # print(f'na count : {pd.isna(data).count()}')
        data = data.drop_duplicates(subset=[column])
        data.reset_index(drop=True, inplace = True)
        
        return data
    
    def drop_useless(self, data, column):
        '''drop na data
        '''
        print(f'na count : {pd.isna(data).count()}')
        data.dropna(inplace = True)
        data.reset_index(drop=True, inplace = True)
        
        return data
    
    def process_trans(self, data, column):
        '''transform op data to 0/1
        '''
        data[column+'_'] = data[column]
        for index_ in range(len(data[column])):
            if pd.isnull(data[column][index_]) or pd.isna(data[column][index_]):
                data[column][index_] = 0
            else:
                data[column][index_] = 1

        data[column] = data[column].astype(int)
        
        return data
    
    def process_outlier(self, data, column):
        # quantile 四分位
        q1 = data[column].quantile(0.25) # upper quantile
        q3 = data[column].quantile(0.75) # lower quantile
        iqr = q3 - q1 # interquartile range 四分位距
        # outlier
        lower_quantile = q1 - (iqr * 3)
        upper_quantile = q3 + (iqr * 3)
        
        data = data[data[column] > lower_quantile]
        data = data[data[column] < upper_quantile]
        
        return data
    





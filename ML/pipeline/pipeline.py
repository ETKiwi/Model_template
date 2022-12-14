# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:17:23 2022

@author: BK
"""
class Pipeline:
    '''Framework
    '''
    def __init__(self, steps):
        self.steps = steps
        
    def run(self, inputs, utils):
        data = None
        for step in self.steps:
            try:
                data = step.process(data, inputs, utils)
                print(f'\nPipeline ->\n\
                      Step_Class: {step.__class__.__name__}\n\
                      type: {type(data)}')
            except Exception as e:
                print('Exception happened:', e)
                break
            




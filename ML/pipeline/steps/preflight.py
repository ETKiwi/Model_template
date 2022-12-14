# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:29:24 2022

@author: BK
"""
# import os
# import glob

from ML.pipeline.steps.step import Step


class Preflight(Step):
    def process(self, data, inputs, utils):
        print('in Preflight')





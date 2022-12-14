# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:30:27 2022

@author: user
"""

import matplotlib.pyplot as plt

class Utils:
    def __init__(self, path): #, data):
        self.path = path
        # self.data = data
        
    def plt_pie(self, data1, data2, label1, label2, title):
        plt.figure(dpi = 400)
    
        plt.pie([data1, data2], 
                labels = [label1, label2],
                # pctdistance = 0.6,
                textprops = {"fontsize" : 12},
                autopct = "%.1f%%")
        plt.title(title)
        plt.axis('equal')
        plt.legend(loc = 'best')
        
        plt.savefig(self.path + title + '.jpg',
                bbox_inches='tight', # remove label space
                pad_inches=0.0)  # remove white background





# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:45:14 2022

@author: user
"""
from abc import ABC
from abc import abstractmethod


class Step(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, data, inputs, utils):
        pass


class StepException(Exception):
    pass




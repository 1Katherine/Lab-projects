#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：learn_hyperopt 
@File ：1.1 The Simplest Case.py
@Author ：Yang
@Date ：2022/2/28 15:25 
'''
from hyperopt import fmin, tpe, hp
best = fmin(fn=lambda x: x ** 2,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=100)
print(best)
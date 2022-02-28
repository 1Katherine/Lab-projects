#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：learn_hyperopt 
@File ：1.2 Attaching Extra Information via the Trials Object.py
@Author ：Yang
@Date ：2022/2/28 15:28 
'''
import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK

def objective(x):
    return {'loss': x ** 2, 'status': STATUS_OK }

best = fmin(objective,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=100)

print(best)
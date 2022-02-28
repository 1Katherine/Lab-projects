#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：learn_hyperopt 
@File ：1.3 The Trials Object.py
@Author ：Yang
@Date ：2022/2/28 15:30 
'''
import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# trials : 一个字典列表，代表了所有关于这次搜索的信息
def objective(x):
    return {
        'loss': x ** 2,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        'other_stuff': {'type': None, 'value': [0, 1, 2]},
        # -- attachments are handled differently
        'attachments':
            {'time_module': pickle.dumps(time.time)}
        }
trials = Trials()
best = fmin(objective,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)

print(best)
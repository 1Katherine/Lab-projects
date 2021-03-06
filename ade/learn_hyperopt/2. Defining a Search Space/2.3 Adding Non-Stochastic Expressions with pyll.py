#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：learn_hyperopt 
@File ：2.3 Adding Non-Stochastic Expressions with pyll.py
@Author ：Yang
@Date ：2022/2/28 15:44 
'''
from hyperopt import hp
import hyperopt.pyll
from hyperopt.pyll import scope

@scope.define
def foo(a, b=0):
     print ('runing foo', a, b)
     return a + b / 2

# -- this will print 0, foo is called as usual.
print (foo(0))

# In describing search spaces you can use `foo` as you
# would in normal Python. These two calls will not actually call foo,
# they just record that foo should be called to evaluate the graph.

space1 = scope.foo(hp.uniform('a', 0, 10))
space2 = scope.foo(hp.uniform('a', 0, 10), hp.normal('b', 0, 1))

# -- this will print an pyll.Apply node
print (space1)

# -- this will draw a sample by running foo()
print (hyperopt.pyll.stochastic.sample(space1))
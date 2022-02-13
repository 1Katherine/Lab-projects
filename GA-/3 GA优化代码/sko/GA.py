#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987


import numpy as np
from .base import SkoBase
from sko.tools import func_transformer
from abc import ABCMeta, abstractmethod
from .operators import crossover, mutation, ranking, selection
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from samples import LHS_sample

logfile = open('./log.txt', 'w')

class GeneticAlgorithmBase(SkoBase, metaclass=ABCMeta):
    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200, prob_mut=0.001,
                 constraint_eq=tuple(), constraint_ueq=tuple()):
        self.func = func_transformer(func)
        assert size_pop % 2 == 0, 'size_pop must be even integer'
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.n_dim = n_dim

        # constraint:
        self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        self.constraint_eq = list(constraint_eq)  # a list of equal functions with ceq[i] = 0
        self.constraint_ueq = list(constraint_ueq)  # a list of unequal constraint functions with c[i] <= 0

        self.Chrom = None
        self.X = None  # shape = (size_pop, n_dim)
        self.Y_raw = None  # shape = (size_pop,) , value is f(x)
        self.Y = None  # shape = (size_pop,) , value is f(x) + penalty for constraint
        self.FitV = None  # shape = (size_pop,)

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []

        self.best_x, self.best_y = None, None

    @abstractmethod
    def chrom2x(self, Chrom):
        pass

    # 根据X 和 func 返回性能和y的值
    def x2y(self):
        print('-------------------- 开始 x2y(self) ----------------------', file = logfile)
        self.Y_raw = self.func(self.X)
        # 如果有约束
        if not self.has_constraint:
            self.Y = self.Y_raw
        # 如果没有约束
        else:
            # constraint
            penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        print('--------------------- 结束 x2y(self) ---------------------', file = logfile)
        return self.Y

    @abstractmethod
    def ranking(self):
        pass

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def mutation(self):
        pass

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.X = self.chrom2x(self.Chrom) # 二进制种群对应的实数值个体
            self.Y = self.x2y()

            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y

    fit = run


class GA(GeneticAlgorithmBase):
    """genetic algorithm

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint
    constraint_ueq : tuple
        unequal constraint
    precision : array_like
        The precision of every variables of func
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    Attributes
    ----------------------
    Lind : array_like
         The num of genes of every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    Examples
    -------------
    https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga.py
    """

    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, constraint_eq, constraint_ueq)

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)

        self.precision = np.array(precision) * np.ones(self.n_dim)  # works when precision is int, float, list or array

        # Lind is the num of genes of every variable of func（segments）

        Lind_raw = np.log2((self.ub - self.lb) / self.precision + 1)

        self.Lind = np.ceil(Lind_raw).astype(int)

        # if precision is integer:
        # if Lind_raw is integer, which means the number of all possible value is 2**n, no need to modify
        # if Lind_raw is decimal, we need ub_extend to make the number equal to 2**n,

        self.int_mode_ = (self.precision % 1 == 0) & (Lind_raw % 1 != 0)

        self.int_mode = np.any(self.int_mode_)

        if self.int_mode:
            self.ub_extend = np.where(self.int_mode_
                                      , self.lb + (np.exp2(self.Lind) - 1) * self.precision
                                      , self.ub)

        self.len_chrom = sum(self.Lind)

        self.crtbp()

    def crtbp(self):
        # create the population
        self.Chrom = np.random.randint(low=0, high=2, size=(self.size_pop, self.len_chrom))
        return self.Chrom

    def gray2rv(self, gray_code):
        # Gray Code to real value: one piece of a whole chromosome
        # input is a 2-dimensional numpy array of 0 and 1.
        # output is a 1-dimensional numpy array which convert every row of input into a real number.
        _, len_gray_code = gray_code.shape
        b = gray_code.cumsum(axis=1) % 2
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        return (b * mask).sum(axis=1) / mask.sum()

    def chrom2x(self, Chrom):
        cumsum_len_segment = self.Lind.cumsum()
        X = np.zeros(shape=(self.size_pop, self.n_dim))
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
            else:
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
            X[:, i] = self.gray2rv(Chrom_temp)
        if self.int_mode:
            X = self.lb + (self.ub_extend - self.lb) * X
            X = np.where(X > self.ub, self.ub, X)
            # the ub may not obey precision, which is ok.
            # for example, if precision=2, lb=0, ub=5, then x can be 5
        else:
            X = self.lb + (self.ub - self.lb) * X
        return X

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_2point_bit
    mutation = mutation.mutation

    def to(self, device):
        '''
        use pytorch to get parallel performance
        '''
        try:
            import torch
            from .operators_gpu import crossover_gpu, mutation_gpu, selection_gpu, ranking_gpu
        except:
            print('pytorch is needed')
            return self

        self.device = device
        self.Chrom = torch.tensor(self.Chrom, device=device, dtype=torch.int8)

        def chrom2x(self, Chrom):
            '''
            We do not intend to make all operators as tensor,
            because objective function is probably not for pytorch
            '''
            Chrom = Chrom.cpu().numpy()
            cumsum_len_segment = self.Lind.cumsum()
            X = np.zeros(shape=(self.size_pop, self.n_dim))
            for i, j in enumerate(cumsum_len_segment):
                if i == 0:
                    Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
                else:
                    Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
                X[:, i] = self.gray2rv(Chrom_temp)

            if self.int_mode:
                X = self.lb + (self.ub_extend - self.lb) * X
                X = np.where(X > self.ub, self.ub, X)
            else:
                X = self.lb + (self.ub - self.lb) * X
            return X

        self.register('mutation', mutation_gpu.mutation). \
            register('crossover', crossover_gpu.crossover_2point_bit). \
            register('chrom2x', chrom2x)

        return self


class RCGA(GeneticAlgorithmBase):
    """real-coding genetic algorithm
    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    prob_cros : float between 0 and 1
        Probability of crossover
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    """

    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 prob_cros=0.9,
                 lb=-1, ub=1,
                 ):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut)
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.prob_cros = prob_cros
        self.crtbp()

    def crtbp(self):
        # create the population, random floating point numbers of 0 ~ 1
        self.Chrom = np.random.random([self.size_pop, self.n_dim])
        return self.Chrom

    def chrom2x(self, Chrom):
        X = self.lb + (self.ub - self.lb) * self.Chrom
        return X

    def crossover_SBX(self):
        '''
        simulated binary crossover
        :param self:
        :return self.Chrom:
        '''
        Chrom, size_pop, len_chrom, Y = self.Chrom, self.size_pop, len(self.Chrom[0]), self.FitV
        for i in range(0, size_pop, 2):

            if np.random.random() > self.prob_cros:
                continue
            for j in range(len_chrom):

                ylow = 0
                yup = 1
                y1 = Chrom[i][j]
                y2 = Chrom[i + 1][j]
                r = np.random.random()
                if r <= 0.5:
                    betaq = (2 * r) ** (1.0 / (1 + 1.0))
                else:
                    betaq = (0.5 / (1.0 - r)) ** (1.0 / (1 + 1.0))

                child1 = 0.5 * ((1 + betaq) * y1 + (1 - betaq) * y2)
                child2 = 0.5 * ((1 - betaq) * y1 + (1 + betaq) * y2)

                child1 = min(max(child1, ylow), yup)
                child2 = min(max(child2, ylow), yup)

                self.Chrom[i][j] = child1
                self.Chrom[i + 1][j] = child2
        return self.Chrom

    def mutation(self):
        '''
        Routine for real polynomial mutation of an individual
        mutation of 0/1 type chromosome
        :param self:
        :return:
        '''
        #
        size_pop, n_dim, Chrom = self.size_pop, self.n_dim, self.Chrom
        for i in range(size_pop):
            for j in range(n_dim):
                r = np.random.random()
                if r <= self.prob_mut:
                    y = Chrom[i][j]
                    ylow = 0
                    yup = 1
                    delta1 = 1.0 * (y - ylow) / (yup - ylow)
                    delta2 = 1.0 * (yup - y) / (yup - ylow)
                    r = np.random.random()
                    mut_pow = 1.0 / (1 + 1.0)
                    if r <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * r + (1.0 - 2.0 * r) * (xy ** (1 + 1.0))
                        deltaq = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy ** (1 + 1.0))
                        deltaq = 1.0 - val ** mut_pow
                    y = y + deltaq * (yup - ylow)
                    y = min(yup, max(y, ylow))
                    self.Chrom[i][j] = y
        return self.Chrom

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover_SBX
    mutation = mutation



class myGA(GeneticAlgorithmBase):
    def __init__(self, initsamples, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, constraint_eq, constraint_ueq)
        self.initsamples = initsamples
        self.rs_sizepop = int(self.size_pop / 2) # 用于随机采样生成的个体
        self.lhs_sizepop = self.size_pop - self.rs_sizepop # 用于其他采样方式生成的个体
        # n_dim = 搜索参数的个数（比如有10个需要优化的重要参数 n_dim = 10）
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.precision = np.array(precision) * np.ones(self.n_dim)  # works when precision is int, float, list or array
        Lind_raw = np.log2((self.ub - self.lb) / self.precision + 1)
        self.Lind = np.ceil(Lind_raw).astype(int)


        self.int_mode_ = (self.precision % 1 == 0) & (Lind_raw % 1 != 0)
        self.int_mode = np.any(self.int_mode_)
        if self.int_mode:
            self.ub_extend = np.where(self.int_mode_
                                      , self.lb + (np.exp2(self.Lind) - 1) * self.precision
                                      , self.ub) # ub_extend = ndarray(10,)
        # 染色体长度（所有变量的基因数量总和） len_chrom = 81
        self.len_chrom = sum(self.Lind)

        # ------------新增代码 start--------------
        self.bitsPower = np.ceil(Lind_raw).astype(int)
        self.realPrecision = np.zeros(shape=(len(self.bitsPower)))
        for i in range(len(self.bitsPower)):
            # 真实精度
            self.realPrecision[i] = (self.ub[i] - self.lb[i]) / (2 ** self.bitsPower[i] - 1)
        # ------------新增代码 end--------------
        # 生成初始种群  Chrom = [[1 1 0... 0 1 1]
        self.crtbp()

    # 产生初始种群（0-1），行为种群个数，列为染色体长度（所有变量的基因数量总和）
    def crtbp(self):
        # create the population
        # 返回一个随机整数，范围从[0,2), 随机数的尺寸为 size_pop * len_chrom (sizePop * 81)
        # Chrom = [[1 1 0... 0 1 1]
        #          [0 0 0... 0 1 0]....
        # ------------注释代码 start--------------
        # self.Chrom = np.random.randint(low=0, high=2, size=(self.size_pop, self.len_chrom))
        # ------------注释代码 end----------------
        # ------------新增代码 start--------------
        # Chrom_rs = np.random.randint(low=0, high=2, size=(self.rs_sizepop, self.len_chrom))
        # lhssamples = self.samples()
        # Chrom_sample = self.numTo2x(lhssamples)
        # # 垂直拼接两个矩阵
        # self.Chrom = np.vstack((Chrom_rs, Chrom_sample))
        # self.Chrom = self.Chrom.astype(np.int32)


        # 针对初始样本越界情况，往往发生在范围表中的上下界与样本实际值不符
        for row, sam in enumerate(self.initsamples):
            for dim in range(self.n_dim):
                if sam[dim] < self.lb[dim]:
                    print('下界：' + str(self.lb))
                    print(sam)
                    raise OverflowError('错误发生在第'+ str(row + 1) +'个样本的第'+ str(dim + 1) +'个维度，变量值'+ str(sam[dim]) +'超出下界'+ str(self.lb[dim]) +'！请检查范围表！')
                if sam[dim] > self.ub[dim]:
                    print('上界：' + str(self.ub))
                    print(sam)
                    raise OverflowError('错误发生在第'+ str(row + 1) +'个样本的第'+ str(dim + 1) +'个维度，变量值'+ str(sam[dim]) +'超出上界'+ str(self.ub[dim]) +'！请检查范围表！')
        Chrom_sample = self.numTo2x(self.initsamples)
        self.Chrom = Chrom_sample.astype(np.int32)
        # ------------新增代码 end--------------
        return self.Chrom


    # ------------新增代码 start--------------


    # 采样实数值样本
    def samples(self):
        self.bounds = list(zip(self.lb, self.ub))
        l = LHS_sample.LHSample(self.n_dim, self.bounds, self.lhs_sizepop)
        lhsample = l.lhs()
        # print('产生的样本lhsample = \n' + str(lhsample))
        for sample in lhsample:
            for dim in range(self.n_dim):
                if self.precision[dim] == 1.0:
                    sample[dim] = round(sample[dim])
        # print('根据精度操作样本lhsample = \n' + str(lhsample))
        return lhsample

    # 样本转二进制种群
    def numTo2x(self, samples):
        self.len_Chrom = self.bitsPower.sum()
        self.Chrom = np.zeros(shape=(len(samples), self.len_Chrom))
        self.cumsum_len_segment = self.bitsPower.cumsum()
        for row, sample in enumerate(samples):
            X = np.zeros(shape=(self.len_Chrom))
            for dim in range(self.n_dim):
                code = self.GetCodeParameter(sample[dim], dim)
                if dim == 0:
                    X[:self.cumsum_len_segment[dim]] = code
                else:
                    X[self.cumsum_len_segment[dim - 1]:self.cumsum_len_segment[dim]] = code
            self.Chrom[row:row + 1, :] = X
        return self.Chrom

    # 计算实数x编码后的二进制值
    def GetCodeParameter(self, x, x_dim):
        # x 对应的十进制数
        codenum = (x - self.lb[x_dim]) / self.realPrecision[x_dim]
        # 十进制转二进制
        mycode = ''
        # 如果二进制的第一位为-(-0b)，则删除前三个元素
        if bin(int(codenum))[0] == '-':
            mycode = bin(int(codenum))[3:]
        # 如果二进制的第一位为0(0b)，则删除前两个元素
        if bin(int(codenum))[0] == '0':
            mycode = bin(int(codenum))[2:]
        while len(mycode) < self.bitsPower[x_dim]:
            # 头部插入0
            mycode = '0' + mycode
        # print(code)
        return np.array(list(mycode))
    # ------------新增代码 end--------------

    # ------------新增代码 start--------------
    def getGrayValue(self, graycode, code_dim):
        X = []
        for code in graycode:
            graycode_str = code.tolist()
            code_str = ''
            for c in graycode_str:
                if c == 0.0:
                    code_str = code_str + '0'
                if c == 1.0:
                    code_str = code_str + '1'
            X.append(int(code_str , 2) * self.realPrecision[code_dim] + self.lb[code_dim])
        return np.array(X)

    def chrom2x(self, Chrom):
        cumsum_len_segment = self.bitsPower.cumsum()
        X = np.zeros(shape=(self.size_pop, self.n_dim))
        # i = 0\1\2\3...9（变量个数）
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                # 取初始种群 Chrom 的前6列 （第一个变量）
                Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
            else:
                # 取 Chorm 的第 cumsum_len_segment[i - 1] 到第 cumsum_len_segment[i]列（其他变量列）
                # 取每一个变量对应的染色体子片段
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
            # 将每一个变量的片段染色体的 0-1 之间的实数值放入返回值X的对应列中，i表示该变量对应的维度下标
            X[:, i] = self.getGrayValue(Chrom_temp, i)
            # 根据精度调整每个变量的值
            for x in X:
                for dim in range(self.n_dim):
                    if self.precision[dim] == 1.0:
                        x[dim] = round(x[dim])
        # print('二进制解码成十进制实数,X =  \n' + str(X[self.lhs_sizepop:,:]))

        return X
    # ------------新增代码 end--------------

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_2point_bit
    mutation = mutation.mutation

'''
    实数编码GA
'''
class myRCGA(GeneticAlgorithmBase):
    def __init__(self, initsamples, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 prob_cros=0.9,
                 lb=-1, ub=1,
                ):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut)
        self.initsamples = initsamples
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.prob_cros = prob_cros
        self.crtbp()

    def crtbp(self):
        # create the population, random floating point numbers of 0 ~ 1
        '''
            注释代码 ： 原本只使用随机采样的实数编码，随机产生（0-1）之间的实数值
            注释时间：2022/1/12 15：01
        '''
        # self.Chrom = np.random.random([self.size_pop, self.n_dim])
        # return self.Chrom
        '''
            新增代码：50% 随机采样，50% lhs采样
            新增时间：2022/1/12 15：01
        '''
        # rs_sizepop = int(self.size_pop / 2)  # 用于随机采样生成的个体
        # lhs_sizepop = self.size_pop - rs_sizepop  # 用于其他采样方式生成的个体
        # # lhs采样
        # bounds = list(zip(self.lb, self.ub))
        # l = LHS_sample.LHSample(self.n_dim, bounds, lhs_sizepop)
        # lhsample = l.lhs()
        # sample_Chrom = self.reversechrom2x(lhsample)
        # # 随机采样
        # rs_Chrom = np.random.random([rs_sizepop, self.n_dim])
        # # 垂直组合两个矩阵
        # self.Chrom = np.vstack((rs_Chrom, sample_Chrom))
        for row, sam in enumerate(self.initsamples):
            for dim in range(self.n_dim):
                if sam[dim] < self.lb[dim]:
                    raise OverflowError('错误发生在第'+ str(row + 1) +'个样本的第'+ str(dim + 1) +'个维度，变量值'+ str(sam[dim]) +'超出下界'+ str(self.lb[dim]) +'！请检查范围表！')
                if sam[dim] > self.ub[dim]:
                    raise OverflowError('错误发生在第'+ str(row + 1) +'个样本的第'+ str(dim + 1) +'个维度，变量值'+ str(sam[dim]) +'超出上界'+ str(self.ub[dim]) +'！请检查范围表！')
        self.Chrom = self.reversechrom2x(self.initsamples)
        return self.Chrom


    # 0-1之间的实数编码计算样本在范围内的真实值
    def chrom2x(self,Chrom):
        X = self.lb + (self.ub - self.lb) * self.Chrom
        return X

    # 其他采样得到的实数值逆向转换成（0 - 1之间）实数编码
    def reversechrom2x(self, X):
        sample_Chrom = (X - self.lb) / (self.ub - self.lb)
        return sample_Chrom

    def crossover_SBX(self):
        '''
        simulated binary crossover
        :param self:
        :return self.Chrom:
        '''
        Chrom, size_pop, len_chrom, Y = self.Chrom, self.size_pop, len(self.Chrom[0]), self.FitV
        for i in range(0, size_pop, 2):
            if np.random.random() > self.prob_cros: # 如果随机数大于交叉概率，才进行交叉概率
                continue
            for j in range(len_chrom):
                ylow = 0
                yup = 1
                y1 = Chrom[i][j]
                y2 = Chrom[i + 1][j]
                r = np.random.random()
                if r <= 0.5:
                    betaq = (2 * r) ** (1.0 / (1 + 1.0))
                else:
                    betaq = (0.5 / (1.0 - r)) ** (1.0 / (1 + 1.0))

                child1 = 0.5 * ((1 + betaq) * y1 + (1 - betaq) * y2)
                child2 = 0.5 * ((1 - betaq) * y1 + (1 + betaq) * y2)

                child1 = min(max(child1, ylow), yup)
                child2 = min(max(child2, ylow), yup)

                self.Chrom[i][j] = child1
                self.Chrom[i + 1][j] = child2
        return self.Chrom

    def mutation(self):
        '''
        Routine for real polynomial mutation of an individual
        mutation of 0/1 type chromosome
        :param self:
        :return:
        '''
        #
        size_pop, n_dim, Chrom= self.size_pop, self.n_dim, self.Chrom
        for i in range(size_pop):
            for j in range(n_dim):
                r = np.random.random()
                # if r <= self.prob_mut:
                if r <= self.prob_mut[j]:# 如果随机数小于变异概率才进行变异操作
                    y = Chrom[i][j]
                    ylow = 0
                    yup = 1
                    delta1 = 1.0 * (y - ylow) / (yup - ylow)
                    delta2 = 1.0 * (yup - y) / (yup - ylow)
                    r = np.random.random()
                    mut_pow = 1.0 / (1 + 1.0)
                    if r <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * r + (1.0 - 2.0 * r) * (xy ** (1 + 1.0))
                        deltaq = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy ** (1 + 1.0))
                        deltaq = 1.0 - val ** mut_pow
                    y = y + deltaq * (yup - ylow)
                    y = min(yup, max(y, ylow))
                    self.Chrom[i][j] = y
        return self.Chrom

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover_SBX
    mutation = mutation



# 用遗传算法来解决TSP（旅行推销员问题）
class GA_TSP(GeneticAlgorithmBase):
    """
    Do genetic algorithm to solve the TSP (Travelling Salesman Problem)
    Parameters
    ----------------
    func : function
        The func you want to do optimal.
        It inputs a candidate solution(a routine), and return the costs of the routine.
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    Attributes
    ----------------------
    Lind : array_like
         The num of genes corresponding to every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    Examples
    -------------
    Firstly, your data (the distance matrix). Here I generate the data randomly as a demo:
    ```py
    num_points = 8
    points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    print('distance_matrix is: \n', distance_matrix)
    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
    ```
    Do GA
    ```py
    from sko.GA import GA_TSP
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=8, pop=50, max_iter=200, Pm=0.001)
    best_points, best_distance = ga_tsp.run()
    ```
    """

    def __init__(self, func, n_dim, size_pop=50, max_iter=200, prob_mut=0.001):
        super().__init__(func, n_dim, size_pop=size_pop, max_iter=max_iter, prob_mut=prob_mut)
        self.has_constraint = False
        self.len_chrom = self.n_dim
        self.crtbp()

    def crtbp(self):
        # create the population
        tmp = np.random.rand(self.size_pop, self.len_chrom)
        self.Chrom = tmp.argsort(axis=1)
        return self.Chrom

    def chrom2x(self, Chrom):
        return Chrom

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_pmx
    mutation = mutation.mutation_reverse

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            # 上一代种群
            Chrom_old = self.Chrom.copy()
            # 根据种群的基因值（二进制0或1）获得参数值 X （ 种群个体数 * 变量个数n_dim )
            self.X = self.chrom2x(self.Chrom)
            # 获得参数值X 对应的性能 Y
            self.Y = self.x2y()

            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # put parent and offspring together and select the best size_pop number of population
            # 将父亲代和子代放在一起，并选择最佳的人口规模_pop数量
            self.Chrom = np.concatenate([Chrom_old, self.Chrom], axis=0)
            # 根据种群基因获得参数值 X
            self.X = self.chrom2x(self.Chrom)
            # 根据参数值 X 计算性能值 Y
            self.Y = self.x2y()
            # 对种群进行排序
            self.ranking()
            # 将数组x中的元素从小到大排列，按顺序返回对应的索引值
            selected_idx = np.argsort(self.Y)[:self.size_pop]
            # 选择排名靠前的 self.size_pop 个个体作为新的种群
            self.Chrom = self.Chrom[selected_idx, :]

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y.copy())
            self.all_history_FitV.append(self.FitV.copy())

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y

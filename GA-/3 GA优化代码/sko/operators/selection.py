import numpy as np

logfile = open('./log.txt', 'w')

def selection_tournament(self, tourn_size=3):
    '''
    Select the best individual among *tournsize* randomly chosen
    individuals,
    :param self:
    :param tourn_size:
    :return:
    '''
    print('selection_tournament', file = logfile)
    FitV = self.FitV
    sel_index = []
    # i = 0 - self.size_pop-1
    for i in range(self.size_pop):
        # aspirants_index = np.random.choice(range(self.size_pop), size=tourn_size)
        # 随机产生 tourn_size 个  [0,size_pop) 整数
        aspirants_index = np.random.randint(self.size_pop, size=tourn_size)
        sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
    self.Chrom = self.Chrom[sel_index, :]  # next generation
    return self.Chrom

# 默认在随机选择的*tournsize*中选择最佳个体，与`selection_tournament`相同，但使用numpy的速度更快
def selection_tournament_faster(self, tourn_size=3):
    '''
    Select the best individual among *tournsize* randomly chosen
    Same with `selection_tournament` but much faster using numpy
    individuals,
    :param self:
    :param tourn_size:
    :return:
    '''
    '''
        选择后续用于交叉的个体，将选择的个体组成一个新的种群，返回选择个体后的新种群Chrom
        概述：随机选择3个个体作为一个team，共产生30个team，从每个team中选择fit值最大的个体作为选择后的个体，
            最终会选择出30个个体作为新的Chrom返回，用于后续交叉计算。
        Chorm = （30，81）：行表示30个个体，列表示该个体对应特征变量的二进制值，根据特征变量的范围大小对应不同数量的二进制列
        1. aspirants_idx（30，3） ：随机产生一个30*3的矩阵，随机产生0-29之间的整数值 （表示个体下标），一行为一个team
        2. aspirants_values（30，3）：FitV（30，）中记录了每一个个体的适应度值，根据个体下标获取该个体的适应度值，一行为一个team
        3. winner（30，） = 0，1，2 ：一行为一个team，对适应度矩阵按照team进行比较，返回最大的那个数值所在的下标
        4. sel_index：i = 0 - 29（winner中的下标），j = 0，1，2 （winner中每一个下标对应的值）
            通过aspirants_idx[i, j]得到该位置对应的原Chrom的个体下标，有的个体可能被选择多次，有的个体不被选择。
        5. 根据最终选择的30个个体的下标，从原Chrom中取出这30个个体，生成新的选择后的Chrom矩阵用于后续的交叉
    
    '''
    print('selection_tournament_faster', file = logfile)
    # sizePop = 30 产生0-29之间的随机整数，aspirants_idx = （30，3）
    aspirants_idx = np.random.randint(self.size_pop, size=(self.size_pop, tourn_size))
    # aspirants_values = （30，3） ，获得每一个aspirants_idx对应的适应度函数值 self.FitV = （30，）
    aspirants_values = self.FitV[aspirants_idx]
    # 对每一列取最大值，获得winner = （30，） winner的值为0、1、2
    winner = aspirants_values.argmax(axis=1)  # winner index in every team
    # sel_index = list(30)，得到30个 0-29之间的下标，表示选择的个体。有的个体没有选择，有的个体选择了多次
    sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]
    # 每一个个体下标对应原种群self.Chrom的一行，列代表该个体的特征变量，取出选择个30个个体作为新的种群
    self.Chrom = self.Chrom[sel_index, :]
    print('selection_tournament ,self.Chrom = ' + str(self.Chrom), file = logfile)
    return self.Chrom


def selection_roulette_1(self):
    '''
    Select the next generation using roulette
    :param self:
    :return:
    '''
    print('selection_roulette_1', file = logfile)
    FitV = self.FitV
    FitV = FitV - FitV.min() + 1e-10
    # the worst one should still has a chance to be selected
    sel_prob = FitV / FitV.sum()
    sel_index = np.random.choice(range(self.size_pop), size=self.size_pop, p=sel_prob)
    self.Chrom = self.Chrom[sel_index, :]
    return self.Chrom


def selection_roulette_2(self):
    '''
    Select the next generation using roulette
    :param self:
    :return:
    '''
    print('selection_roulette_2', file = logfile)
    FitV = self.FitV
    FitV = (FitV - FitV.min()) / (FitV.max() - FitV.min() + 1e-10) + 0.2
    # the worst one should still has a chance to be selected
    sel_prob = FitV / FitV.sum()
    sel_index = np.random.choice(range(self.size_pop), size=self.size_pop, p=sel_prob)
    self.Chrom = self.Chrom[sel_index, :]
    return self.Chrom


import numpy as np

logfile = open('./log.txt', 'w')

def mutation(self):
    '''
    mutation of 0/1 type chromosome
    faster than `self.Chrom = (mask + self.Chrom) % 2`
    :param self:
    :return:
    '''
    '''
        size_pop = 30, len_chrom = 81, prob_mut = 0.01
        mask = (30,81) : true/false 
            如果随机值 < 0.01,true
            如果随机值 < 0.01,false
        开始变异操作
        self.Chrom ^= mask
        注释代码：2022/1/11 19:54
    '''
    # print('mutation', file = logfile)
    # # rand返回0，1之间的随机值，取值范围是[0,1)，不包括1
    # mask = (np.random.rand(self.size_pop, self.len_chrom) < self.prob_mut)
    # # 异或 （相同为0不同为1）
    # self.Chrom ^= mask
    # return self.Chrom
    '''
        新增代码：2022/1/11 19:54
            原传入prob_mut = 0.01，现在传入pro_mub = [0.01,0.02,0.1,....],根据参数的重要性对每个参数进行变异
            必要要求：参数重要性的值必须在（0，1）之间 ，pro_mub = list(self.n_dim)
    '''
    mask = np.full((self.size_pop, self.len_chrom), False, dtype=bool)
    # mask = np.zeros(shape=(self.size_pop, self.len_chrom))
    cumsum_len_segment = self.Lind.cumsum()
    d = dict(zip(cumsum_len_segment, self.prob_mut))
    for i, segment in enumerate(d):
        if i == 0:
            # 取初始种群 Chrom 的前6列 （第一个变量）
            mask[:, :cumsum_len_segment[0]] = np.random.rand(self.size_pop, cumsum_len_segment[0]) < d[segment]
        else:
            # 取 Chorm 的第 cumsum_len_segment[i - 1] 到第 cumsum_len_segment[i]列（其他变量列）
            # 取每一个变量对应的染色体子片段
            size = cumsum_len_segment[i] - cumsum_len_segment[i - 1]
            mask[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]] = np.random.rand(self.size_pop, size) < d[segment]
    self.Chrom ^= mask
    return self.Chrom

def mutation_TSP_1(self):
    '''
    every gene in every chromosome mutate
    :param self:
    :return:
    '''
    print('mutation_TSP_1', file = logfile)
    for i in range(self.size_pop):
        for j in range(self.n_dim):
            if np.random.rand() < self.prob_mut:
                n = np.random.randint(0, self.len_chrom, 1)
                self.Chrom[i, j], self.Chrom[i, n] = self.Chrom[i, n], self.Chrom[i, j]
    return self.Chrom


def swap(individual):
    print('swap', file = logfile)
    n1, n2 = np.random.randint(0, individual.shape[0] - 1, 2)
    if n1 >= n2:
        n1, n2 = n2, n1 + 1
    individual[n1], individual[n2] = individual[n2], individual[n1]
    return individual


def reverse(individual):
    '''
    Reverse n1 to n2
    Also called `2-Opt`: removes two random edges, reconnecting them so they cross
    Karan Bhatia, "Genetic Algorithms and the Traveling Salesman Problem", 1994
    https://pdfs.semanticscholar.org/c5dd/3d8e97202f07f2e337a791c3bf81cd0bbb13.pdf
    '''
    print('reverse', file = logfile)
    n1, n2 = np.random.randint(0, individual.shape[0] - 1, 2)
    if n1 >= n2:
        n1, n2 = n2, n1 + 1
    individual[n1:n2] = individual[n1:n2][::-1]
    return individual


def transpose(individual):
    print('transpose', file = logfile)
    # randomly generate n1 < n2 < n3. Notice: not equal
    n1, n2, n3 = sorted(np.random.randint(0, individual.shape[0] - 2, 3))
    n2 += 1
    n3 += 2
    slice1, slice2, slice3, slice4 = individual[0:n1], individual[n1:n2], individual[n2:n3 + 1], individual[n3 + 1:]
    individual = np.concatenate([slice1, slice3, slice2, slice4])
    return individual


def mutation_reverse(self):
    '''
    Reverse
    :param self:
    :return:
    '''
    print('mutation_reverse', file = logfile)
    for i in range(self.size_pop):
        if np.random.rand() < self.prob_mut:
            self.Chrom[i] = reverse(self.Chrom[i])
    return self.Chrom


def mutation_swap(self):
    print('mutation_swap', file = logfile)
    for i in range(self.size_pop):
        if np.random.rand() < self.prob_mut:
            self.Chrom[i] = swap(self.Chrom[i])
    return self.Chrom

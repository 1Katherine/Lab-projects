import numpy as np

__all__ = ['crossover_1point', 'crossover_2point', 'crossover_2point_bit', 'crossover_pmx', 'crossover_2point_prob']

logfile = open('./log.txt', 'w')

def crossover_1point(self):
    print('crossover_1point', file = logfile)
    Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
    for i in range(0, size_pop, 2):
        n = np.random.randint(0, self.len_chrom)
        # crossover at the point n
        seg1, seg2 = self.Chrom[i, n:].copy(), self.Chrom[i + 1, n:].copy()
        self.Chrom[i, n:], self.Chrom[i + 1, n:] = seg2, seg1
    return self.Chrom


def crossover_2point(self):
    print('crossover_2point', file = logfile)
    Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
    for i in range(0, size_pop, 2):
        n1, n2 = np.random.randint(0, self.len_chrom, 2)
        if n1 > n2:
            n1, n2 = n2, n1
        # crossover at the points n1 to n2
        seg1, seg2 = self.Chrom[i, n1:n2].copy(), self.Chrom[i + 1, n1:n2].copy()
        self.Chrom[i, n1:n2], self.Chrom[i + 1, n1:n2] = seg2, seg1
    return self.Chrom

# 默认使用这个方法
def crossover_2point_bit(self):
    '''
    3 times faster than `crossover_2point`, but only use for 0/1 type of Chrom
    :param self:
    :return:
    '''
    '''
        对选择后的种群Chrom进行交叉操作
        1. half_size_pop = 种群个体数/2
        2. Chrom1为原种群Chrom的前 half_size_pop 个个体矩阵， Chrom2为剩下的个体矩阵
        3. mask = （30，81）的全为0的矩阵
        4. i = 0 - （half_size_pop - 1），对于mask的每一行，随机使得每一行的n1 - （n2-1）列的二进制值为1
        开始交叉计算
        5. mask2 = (Chrom1 ^ Chrom2) & mask
        6. Chrom1 ^= mask2
        7. Chrom2 ^= mask2
    
    '''
    print('crossover_2point_bit', file = logfile)
    # Chorm = (30,81), size_pop = 30, len_chrom = 81
    Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
    # half_size_pop = 15 (选择种群中的一半个体）
    half_size_pop = int(size_pop / 2)
    # Chrom1 = (15,81) 前15个个体, Chrom2 = (15,81)后15个个体
    Chrom1, Chrom2 = Chrom[:half_size_pop], Chrom[half_size_pop:]
    # mask =  (15,81)的零矩阵
    mask = np.zeros(shape=(half_size_pop, len_chrom), dtype=int)
    # i = 0 - 14 ，使第i行的n1 - （n2-1）的二进制位为1
    for i in range(half_size_pop):
        n1, n2 = np.random.randint(0, self.len_chrom, 2) # 在 0-80 之间产生两个随机的整数
        if n1 > n2:
            n1, n2 = n2, n1 # 限制 n1 < n2
        mask[i, n1:n2] = 1  # mask 的第i行，n1-n2全设置为1
    mask2 = (Chrom1 ^ Chrom2) & mask
    Chrom1 ^= mask2
    Chrom2 ^= mask2
    return self.Chrom   # Chrom1，Chrom2变化，Chrom自然变化


def crossover_2point_prob(self, crossover_prob):
    '''
    2 points crossover with probability
    '''
    print('crossover_2point_prob', file = logfile)
    Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
    for i in range(0, size_pop, 2):
        if np.random.rand() < crossover_prob:
            n1, n2 = np.random.randint(0, self.len_chrom, 2)
            if n1 > n2:
                n1, n2 = n2, n1
            seg1, seg2 = self.Chrom[i, n1:n2].copy(), self.Chrom[i + 1, n1:n2].copy()
            self.Chrom[i, n1:n2], self.Chrom[i + 1, n1:n2] = seg2, seg1
    return self.Chrom


# def crossover_rv_3(self):
#     Chrom, size_pop = self.Chrom, self.size_pop
#     i = np.random.randint(1, self.len_chrom)  # crossover at the point i
#     Chrom1 = np.concatenate([Chrom[::2, :i], Chrom[1::2, i:]], axis=1)
#     Chrom2 = np.concatenate([Chrom[1::2, :i], Chrom[0::2, i:]], axis=1)
#     self.Chrom = np.concatenate([Chrom1, Chrom2], axis=0)
#     return self.Chrom


def crossover_pmx(self):
    '''
    Executes a partially matched crossover (PMX) on Chrom.
    For more details see [Goldberg1985]_.

    :param self:
    :return:

    .. [Goldberg1985] Goldberg and Lingel, "Alleles, loci, and the traveling
   salesman problem", 1985.
    '''
    print('crossover_pmx', file = logfile)
    Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
    for i in range(0, size_pop, 2):
        Chrom1, Chrom2 = self.Chrom[i], self.Chrom[i + 1]
        cxpoint1, cxpoint2 = np.random.randint(0, self.len_chrom - 1, 2)
        if cxpoint1 >= cxpoint2:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1 + 1
        # crossover at the point cxpoint1 to cxpoint2
        pos1_recorder = {value: idx for idx, value in enumerate(Chrom1)}
        pos2_recorder = {value: idx for idx, value in enumerate(Chrom2)}
        for j in range(cxpoint1, cxpoint2):
            value1, value2 = Chrom1[j], Chrom2[j]
            pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
            Chrom1[j], Chrom1[pos1] = Chrom1[pos1], Chrom1[j]
            Chrom2[j], Chrom2[pos2] = Chrom2[pos2], Chrom2[j]
            pos1_recorder[value1], pos1_recorder[value2] = pos1, j
            pos2_recorder[value1], pos2_recorder[value2] = j, pos2

        self.Chrom[i], self.Chrom[i + 1] = Chrom1, Chrom2
    return self.Chrom

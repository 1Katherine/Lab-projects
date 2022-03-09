import warnings
import pandas as pd
import numpy as np
import os
import time

from .target_space import TargetSpace, _hashable
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger
from .util import UtilityFunction, acq_max, ensure_rng
from .Snet import train
from .configuration import parser
args = parser.parse_args()

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

# 标准化
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录,比如/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI
father_path = os.path.abspath(os.path.dirname(os.path.dirname(current_path)) + os.path.sep + ".")


class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback is None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)


class BayesianOptimization(Observable):
    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None):
        """"""
        self._random_state = ensure_rng(random_state)
        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)
        # queue
        self._queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            # self._space.probe(params)
            # self.dispatch(Events.OPTIMIZATION_STEP)
            self.flag = True
            if self._space.probe(params):
                self.flag = True
            else:
                self.flag = False
            self.dispatch(Events.OPTIMIZATION_STEP)
            return self.flag

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # self._gp.fit(self._space.params, self._space.target)
            # ---------------------- 新增：参数标准化后建立GP模型 start -----------------------
            train_X_temp = standardization(np.array(self._space.params))
            train_y_temp = np.array(self._space.target)
            print('标准化前的train_x = \n' + str(np.array(self._space.params)))
            print('标准化后的train_x = \n' + str(train_X_temp))

            result_X = np.zeros(shape=(train_X_temp.shape[0],train_X_temp.shape[1]))
            # 对每一列进行标准化
            for col in range(train_X_temp.shape[1]):
                result_X[:,col] = standardization(train_X_temp[:,col])
            print('测试：每一列进行标准化，标准化后的train_x = \n' + str(result_X))

            train_X_temp[np.isnan(train_X_temp)] = 0
            print('标准化时如果存在分母（sigma）为0的情况，返回nan。此时将所有nan替换为0 \n' + str(train_X_temp))

            print('train_y = \n' + str(np.array(self._space.target)))
            y_df = pd.DataFrame(train_y_temp)
            x_df = pd.DataFrame(train_X_temp)
            print('train_X_temp info:')
            print(x_df.info())
            print('train_y_temp info:')
            print(y_df.info())
            self._gp.fit(train_X_temp, train_y_temp)

            # ---------------------- 新增：参数标准化后建立GP模型 end -----------------------

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )

        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)
        '''
            # 注释代码：随机生成样本代码(随机抽样一次只生成一个样本）
            # 2021/12/29 19:17
        '''
        # for _ in range(init_points):
        #     self._queue.add(self._space.random_sample())

        print('------------使用lhs生成初始样本点------------')
        lhsample = self._space.lhs_sample(init_points)
        for l in lhsample:
            self._queue.add(l.ravel())


    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)

        # 记录搜索算法开始时间
        start_time = time.time()

        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay)

        self.Max_time = -100000

        print('self._space.keys = ' + str(self._space.keys))
        params_list = []
        for param in self._space.keys:
            params_list.append(param)
        params_list.append('runtime')

        # m 保存当前所有样本的df
        m = pd.DataFrame(columns=params_list)

        # 运行随机样本，保存在df m中
        while not self._queue.empty:
            x_probe = next(self._queue)
            self.probe(x_probe, lazy=False)
            # 获取该随机样本的执行时间
            x = self._space._as_array(x_probe)
            print('self._space._cache[_hashable(x)] \t' + str(self._space._cache[_hashable(x)]))
            target = self._space._cache[_hashable(x)]
            # 随机样本和执行时间存入m dataframe中
            sample = x.tolist()
            sample.append(target)
            # config.append(sample)
            print('随机采样：config\n' + str(sample))
            n = pd.DataFrame(data=[sample], columns=params_list)
            print('随机采样：存储当前样本的df\n' + str(n))
            m = m.append(n, ignore_index=True)

        # 取随机样本中的最优样本 并训练GAN
        m = m.sort_values('runtime', ascending=False).reset_index(drop=True)
        bestconfig = m.iloc[:1, :-1]


        self.getBestSample_trainGAN(bestconfig, params_list, m, args.gan_initpoints)


        iteration = 0
        while iteration < n_iter:
            print('key = \n' + str(self._space._keys))
            print('bounds = \n' + str(self._space.bounds))
            print('before probe, param.shape = ' + str(self._space.params.shape))
            print('before probe, target = ' + str(self._space.target.shape))
            util.update_params()
            x_probe = self.suggest(util)
            iteration += 1
            if self.probe(x_probe, lazy=False):
                print('x_probe = ' + str(x_probe))
            else:
                print(str(iteration) + '\titeration：执行时间超过' + str(-self.Max_time) + ' s，需要使用gan生成一个样本 \t')
                print('选择当前样本空间的最优配置传给GAN，GAN生成一个配置来运行')
                # test
                params = self._space.params
                runtime = np.array([self._space.target]).T
                inits = np.hstack([params, runtime])
                train_df = pd.DataFrame(inits, columns=params_list)
                # 取随机样本中的最优样本 并训练GAN
                train_df = train_df.sort_values('runtime',ascending =False).reset_index(drop=True)
                bestconfig = train_df.iloc[:1, :-1]

                self.getBestSample_trainGAN(bestconfig, params_list, train_df, 1)


            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        # 记录搜索算法结束时间
        end_time = time.time()
        print(str(int(end_time - start_time)) + 's')  # 秒级时间戳
        self.dispatch(Events.OPTIMIZATION_END)

    def getBestSample_trainGAN(self, bestconfig, params_list, m, num):
        print('bestconfig\n' + str(bestconfig))
        # dataset 保存所有rs样本和gan样本
        dataset = pd.DataFrame(columns=params_list)

        first_time = time.time()
        generate_data = train(bestconfig, first_time, args)
        print(generate_data)
        # 从GAN中选出第一个样本，并运行，保存在df m中
        for i in range(num):
            config = generate_data.iloc[i].tolist()
            # --------- 判断是否越界 ------------
            print('参数和范围为\n' + str(self._space.keys) + "\n" + str(self._space.bounds))
            for i, bound in enumerate(self._space.bounds):
                print('conf为:' + str(self._space.keys[i]) + ' 范围为 = ' + str(bound))
                if config[i] < bound[0]:
                    print(str(self._space.keys[i]) + "越界, 原值为 " + str(config[i]))
                    config[i] = bound[0]
                    print('越界处理后的值为 ' + str(config[i]))
                if config[i] > bound[1]:
                    print(str(self._space.keys[i]) + "越界, 原值为 " + str(config[i]))
                    config[i] = bound[1]
                    print('越界处理后的值为 ' + str(config[i]))
            # --------- 判断是否越界 ------------
            self.probe(config, lazy=False)
            # 获取该样本的执行时间
            try:
                x = self._space._as_array(config)
                target = self._space._cache[_hashable(x)]
                config.append(target)
                print('GAN采样：config\n' + str(config))
                n = pd.DataFrame(data=[config], columns=params_list)
                print('GAN采样：存储当前样本的df\n' + str(n))
                print(n)
                m = m.append(n, ignore_index=True)
                print(m)
            except KeyError:
                print('getBestSample_trainGAN:执行时间超过' + str(-self.Max_time) + ' s，需要使用gan生成一个样本 \t')
        dataset = dataset.append(m, ignore_index=True)
        dataset.to_csv(father_path + '/dataset/dataset_' + str(num) + '_' + str(time.time()) +'.csv')
        print('初始样本点个数：' + str(dataset.shape[0]))

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)

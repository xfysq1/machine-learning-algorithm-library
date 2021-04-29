import numpy as np

class aim_func_GA():

    def aim_func(p):#GA的实例函数
        '''
        This function has plenty of local minimum, with strong shocks
        global minimum at (0,0) with value 0
        '''
        x1, x2 = p
        x = np.square(x1) + np.square(x2)
        return 0.5 + (np.sin(x) - 0.5) / np.square(1 + 0.001 * x)

    constraint_eq = []

    constraint_ueq = []

class aim_func_DE():

    def aim_func_DE(p):#DE的实例函数
        x1, x2, x3 = p
        return x1 ** 2 + x2 ** 2 + x3 ** 2

    constraint_eq = [
        lambda x: 1 - x[1] - x[2]
    ]

    constraint_ueq = [
        lambda x: 1 - x[0] * x[1],
        lambda x: x[0] * x[1] - 5
    ]

class aim_func_PSO:

    def aim_func(x):#PSO的实例函数
        x1, x2, x3 = x
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2








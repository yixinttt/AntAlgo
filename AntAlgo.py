import numpy as np
import matplotlib.pyplot as plt

class Path(object):
    def __init__(self, A):  # A为起始任务
        self.path = [A]

    def add_path(self, B):  # 追加路径信息，方便计算整体目标代价
        self.path.append(B)

class ACOAlgo(object):
    def __init__(self, constraintMatrix, c, taskTime, sd, hazardous, demand, log,
                 ant_num=50, maxIter=30, alpha=1, beta=5, rho=0.1, Q=1, q0=0.1, q1=0.9):
        self.constraintMatrix = constraintMatrix
        self.c = c
        self.taskTime = taskTime
        self.sd = sd
        self.hazardous = hazardous
        self.demand = demand
        self.ants_num = ant_num  # 蚂蚁个数
        self.maxIter = maxIter  # 蚁群最大迭代次数
        self.alpha = alpha  # 信息启发式因子
        self.beta = beta  # 期望启发式因子
        self.rho = rho  # 信息素挥发速度
        self.Q = Q  # 信息素强度
        ###########################
        self.deal_data()  # 提取所有任务信息
        ###########################
        self.path_seed = np.zeros(self.ants_num).astype(int)  # 记录一次迭代过程中每个蚂蚁的初始任务下标
        self.ants_info = np.zeros((self.maxIter, self.ants_num))  # 记录每次迭代后所有蚂蚁的路径长度信息
        self.q0 = q0 #论文中值, 文中没有给定默认值, 该值应该属于经验值, 本处默认设为0.1
        self.q1 = q1 #论文中值, 文中没有给定默认值, 该值应该属于经验值, 本处默认设为0.9
        self.log = log #打日志
        self.bestF1 = [] #用于绘图
        self.bestF2 = []
        self.bestF3 = []
        self.bestF4 = []

    def deal_data(self):
        self.tasks_num = len(self.constraintMatrix)  # 1. 获取任务个数
        self.tasks = list(item for item in range(1, self.tasks_num+1))  # 2. 构建任务列表
        self.phero_mat = np.ones((self.tasks_num, self.tasks_num))  # 3. 初始化信息素矩阵
        self.eta_mat = self.calEtaMat()  # 4. 初始化启发函数矩阵

    def solve(self):
        iterNum = 0  # 当前迭代次数
        while iterNum < self.maxIter:
            self.random_seed()  # 使整个蚁群产生随机的起始点
            delta_phero_mat = np.zeros((self.tasks_num, self.tasks_num))  # 初始化每次迭代后信息素矩阵的增量
            self.log.logger.info("iterator number: " + str(iterNum))
            ##########################################################################
            for i in range(self.ants_num):
                task_index = self.path_seed[i]  # 每只蚂蚁访问的第一个任务下标
                ant_path = Path(self.tasks[task_index])  # 记录每只蚂蚁访问过的任务
                assignedTask = [task_index]  # 记录每只蚂蚁访问过的任务下标，禁忌任务下标列表
                self.log.logger.info("ant No.: " + str(i) + ", begin task: "
                                     + str(task_index) + ", path: " + str(ant_path.path))
                toAssignedTask = self.getAvailableTask(assignedTask) #获取蚂蚁下一步可走的任务(必须满足约束的任务)
                self.log.logger.info("to assigned tasks: " + str(toAssignedTask))
                for j in range(self.tasks_num - 1):  # 对余下的任务进行访问, 该处循环需要将这只蚂蚁的路径全部计算出来
                    up_proba = np.zeros(self.tasks_num - len(assignedTask))  # 初始化状态迁移概率的分子
                    for k in range(len(toAssignedTask)): #根据蚁群算法计算选择下一任务的概率
                        up_proba[k] = np.power(self.phero_mat[task_index][toAssignedTask[k]], self.alpha) * \
                                      np.power(self.eta_mat[toAssignedTask[k]], self.beta)
                    self.log.logger.info("up_proba: " + str(up_proba))
                    proba = up_proba / sum(up_proba)  # 每条可能子路径上的状态迁移概率
                    self.log.logger.info("proba: " + str(proba))
                    while True:  # 提取出下一个任务的下标
                        random_num = np.random.rand() #生成随机数
                        if random_num <= self.q0: #根据论文描述, 通过判断随机数与q0,q1的大小关系,决策下一任务
                            maxProba = max(up_proba) #小于q0取值最大的任务
                            task_index2 = toAssignedTask[np.where(up_proba == maxProba)[0][0]]
                            break
                        elif random_num > self.q1: #大于q1随机选取一个任务
                            ranI = np.random.randint(0, len(toAssignedTask))
                            task_index2 = toAssignedTask[ranI]
                            break
                        else: #否则, 取概率大于随机值的一个任务
                            index_need = np.where(proba > random_num)[0]
                            if len(index_need) > 0:
                                task_index2 = toAssignedTask[index_need[0]]
                                break
                    ant_path.add_path(self.tasks[task_index2]) #该任务添加到路径
                    assignedTask.append(task_index2)
                    toAssignedTask = self.getAvailableTask(assignedTask)
                    task_index = task_index2
                    self.log.logger.info("ant_path: " + str(ant_path.path))
                f1 = self.calF1(ant_path.path)
                self.ants_info[iterNum][i] = self.calF2(f1, self.c)
                if iterNum == 0 and i == 0:  # 完成对最佳路径任务的记录
                    self.best_tasks = ant_path.path #第一轮迭代的第一只蚂蚁的路径设置最优解
                else:
                    if self.cmpOptiObj(ant_path.path, self.best_tasks) : self.best_tasks = ant_path.path #根据目标值判断当前解是否较优
                for l in range(self.tasks_num - 1): #计算该只蚂蚁对信息素增量的贡献(即蚂蚁走过的地方会释放信息素,蚂蚁走依次会叠加一次)
                    delta_phero_mat[assignedTask[l]][assignedTask[l + 1]] += self.Q / self.ants_info[iterNum][i]

            bestF1 = self.calF1(self.best_tasks)
            self.bestF1.append(len(bestF1))
            self.bestF2.append(self.calF2(bestF1, self.c))
            self.bestF3.append(self.calF3(self.best_tasks))
            self.bestF4.append(self.calF4(self.best_tasks))
            self.update_phero_mat(delta_phero_mat)  # 更新信息素矩阵
            iterNum += 1
        return self.best_tasks

    def calWorkTimeSum(self, bestPath):
        f1 = self.calF1(bestPath)
        total = 0
        for f in f1:
            total += f[0]

        return total

    def getAvailableTask(self, assignedTask):
        remainTask = list(set(range(self.tasks_num)) - set(assignedTask))
        toAssignedTask = set()
        for i in remainTask:
            preList = set()
            tmpPreList = {i}
            while 1:
                taskIdList = set()
                for pre in tmpPreList:
                    index = 0
                    for val in self.constraintMatrix[:, pre]:
                        if val == 1:
                            taskIdList.add(index)
                            preList.add(index)
                        index += 1
                if not taskIdList:
                    break
                else:
                    tmpPreList = taskIdList
            if preList.issubset(assignedTask) or len(preList) == 0:
                toAssignedTask = toAssignedTask | {i}

        return list(toAssignedTask)

    def cmpOptiObj(self, currentIterPath, bestPath):
        sgf1 = self.calF1(currentIterPath)
        scf1 = self.calF1(bestPath)
        if len(sgf1) != len(scf1):
            return len(sgf1) < len(scf1)

        sgf2 = self.calF2(sgf1, self.c)
        scf2 = self.calF2(scf1, self.c)
        if sgf2 != scf2:
            return sgf2 < scf2

        sgf3 = self.calF3(currentIterPath)
        scf3 = self.calF3(bestPath)
        if sgf3 != scf3:
            return sgf3 < scf3

        sgf4 = self.calF4(currentIterPath)
        scf4 = self.calF4(bestPath)
        return sgf4 < scf4

    def update_phero_mat(self, delta):#更新信息素矩阵
        self.phero_mat = (1 - self.rho) * self.phero_mat + delta

    def random_seed(self):
        #获取起始任务列表
        startTaskList = self.getStartTaskList()
        print("startTaskList", startTaskList)
        startTaskNum = len(startTaskList)
        # 产生随机的起始点，尽量保证所有蚂蚁的起始点不同
        if self.ants_num <= startTaskNum:  # 蚂蚁数 <= 任务数
            self.path_seed[:] = np.random.permutation(startTaskList)[:self.ants_num]
        else:  # 蚂蚁数 > 任务数
            self.path_seed[:startTaskNum] = np.random.permutation(startTaskList)
            temp_index = startTaskNum
            while temp_index + startTaskNum <= self.ants_num:
                self.path_seed[temp_index:temp_index + startTaskNum] = np.random.permutation(startTaskList)
                temp_index += startTaskNum
            temp_left = self.ants_num % startTaskNum
            if temp_left != 0:
                self.path_seed[temp_index:] = np.random.permutation(startTaskList)[:temp_left]

    def getStartTaskList(self):
        startTaskList = []
        for i in range(self.tasks_num):
            index = 0
            for val in self.constraintMatrix[:, i]:
                if val == 0:
                    index += 1
            if index == self.tasks_num:
                startTaskList.append(i)
        return startTaskList

    def getSucceedTask(self):
        succeedList = np.zeros(len(self.constraintMatrix)).astype(int)
        for i in range(len(self.constraintMatrix)):
            tmpPreList = {i+1}
            tmpSucList = {i+1}
            while 1:
                taskIdList = set()
                for pre in tmpPreList:
                    index = 0
                    for val in self.constraintMatrix[pre-1]:
                        index += 1
                        if val == 1:
                            taskIdList.add(index)
                            tmpSucList.add(index)
                if not taskIdList:
                    break
                else:
                    tmpPreList = taskIdList
            succeedList[i] = len(tmpSucList)-1
        print(succeedList)
        return succeedList

    def calEtaMat(self): #根据论文中给定的公式,计算visibility etaj of task j
        succeedList = self.getSucceedTask() #得到每个任务的后继数量
        etaMat = np.zeros(len(succeedList))
        for i in range(len(succeedList)): #用任务时间/站时间+任务i的后继数量/最后后继数量
            etaMat[i] = self.taskTime[i] / self.c + succeedList[i] / max(succeedList)

        return etaMat


    def calF1(self, Sg):
        '''
        :param Sg: 当前迭代解
        :return 各站工作时间
        '''
        stationWorkTime = []
        tmpTime = 0
        times = 0
        taskList = []
        for i in range(len(Sg)):
            times += self.taskTime[Sg[i] - 1]
            if self.checkSeqDep(self.sd[:, Sg[i] - 1]):
                sdtime = self.getSeqDepTime(Sg[0:i], self.sd[:, Sg[i] - 1])
                times += sdtime
            if tmpTime + times <= self.c:
                taskList.append(Sg[i])
                tmpTime += times
                times = 0
                continue
            else:
                stationWorkTime.append([tmpTime, taskList])
                tmpTime = times
                times = 0
                taskList = []
                taskList.append(Sg[i])
        stationWorkTime.append([tmpTime, taskList])
        # print("stationWokrTime: ", stationWorkTime)
        return stationWorkTime

    def checkSeqDep(self, sdList):
        for s in sdList:
            if s != 0:
                return True
        return False

    def getSeqDepTime(self, sgSublist, sdList):
        sdtime = 0
        for i in range(len(sdList)):
            if (i + 1) not in sgSublist:
                sdtime += sdList[i]
        return sdtime

    def calF2(self, stationWorkTime, c):
        '''
        :param stationWorkTime: 各站工作时间
        :return: f2
        '''
        f2 = 0
        for t in stationWorkTime:
            f2 += np.power(c - t[0], 2)

        return f2

    def calF3(self, Sg):
        '''
        :param Sg: 当前迭代解
        :return: f3
        '''
        f3 = 0
        for i in range(0, len(Sg)):
            f3 += (i + 1) * self.hazardous[Sg[i] - 1]

        return f3

    def calF4(self, Sg):
        '''
        :param Sg: 当前迭代解
        :return: f4
        '''
        f4 = 0
        for i in range(0, len(Sg)):
            f4 += (i + 1) * self.demand[Sg[i] - 1]

        return f4

    def display(self, data, label, outName):  # 数据可视化展示
        plt.figure(figsize=(8, 5))
        plt.plot(list(x for x in range(self.maxIter)), list(y for y in data), label=label)
        plt.xlabel('Iteration')
        plt.ylabel('optimization objective')
        plt.legend()
        plt.savefig(outName, dpi=500)
        plt.show()
        plt.close()

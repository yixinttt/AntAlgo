# -*- coding: utf-8 -*-
import numpy as np
from AntAlgo import ACOAlgo
from Logger import Logger
import csv

def getConstraintMatrix(): #读取任务间约束数据
    csv_reader = csv.reader(open("data/constraintMat.csv"))
    conMat = []
    index = 0
    for row in csv_reader:
        if index == 0:
            index += 1
            continue
        arr = list(map(int, row[1:]))
        conMat.append(arr)
    return np.array(conMat)

def getTaskTime(): #读取任务拆卸时间
    csv_reader = csv.reader(open("data/taskTime.csv"))
    index = 0
    time = []
    for row in csv_reader:
        if index == 0:
            index += 1
            continue
        time = list(map(int, row))

    return time

def getSeqDependencies(): #读取任务依赖消耗时间
    csv_reader = csv.reader(open("data/seqDependencies.csv"))
    sdMat = []
    index = 0
    for row in csv_reader:
        if index == 0:
            index += 1
            continue
        arr = list(map(int, row[1:]))
        sdMat.append(arr)
    return np.array(sdMat)

def getHazardous(): #读取拆卸危险度
    csv_reader = csv.reader(open("data/hazardous.csv"))
    index = 0
    hazardous = []
    for row in csv_reader:
        if index == 0:
            index += 1
            continue
        hazardous = list(map(int, row))
    print(hazardous)
    return hazardous

def getDemand(): #读取任务需求量
    csv_reader = csv.reader(open("data/demand.csv"))
    index = 0
    demand = []
    for row in csv_reader:
        if index == 0:
            index += 1
            continue
        demand = list(map(int, row))
    print(demand)
    return demand


def writeResult():
    with open("result.txt", 'w') as f:
        f.write("best solution: " + str(bestSolution) + "\n")
        f.write("need work station number: " + str(len(f1)) + "\n")
    index = 1
    for f_index in f1:
        log.logger.info(
            "station No.: " + str(index) + ", station work times: " + str(f_index[0]) + ", taskId: " + str(f_index[1]))
        with open('result.txt', 'a') as f:
            f.write("station No.: " + str(index) + ", station work times: " + str(f_index[0]) + ", taskId: " + str(
                f_index[1]) + "\n")
        index += 1


if __name__ == '__main__':
    log = Logger('all.log', level='debug')
    log.logger.info("Ant algo begin...")
    c = 40
    acoAlgo = ACOAlgo(getConstraintMatrix(), c, getTaskTime(), getSeqDependencies(), getHazardous(), getDemand(), log)
    bestSolution = acoAlgo.solve()
    f1 = acoAlgo.calF1(bestSolution)
    log.logger.info("Ant algo end...")
    log.logger.info("best solution: " + str(bestSolution))
    log.logger.info("need work station number: " + str(len(f1)) + "\n")
    writeResult()

    acoAlgo.display(acoAlgo.bestF1, 'f1', 'f1.png')
    acoAlgo.display(acoAlgo.bestF2, 'f2', 'f2.png')
    acoAlgo.display(acoAlgo.bestF3, 'f3', 'f3.png')
    acoAlgo.display(acoAlgo.bestF4, 'f4', 'f4.png')




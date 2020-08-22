import sys
import numpy as np
from random import sample
from random import randint,random
from math import pow,log,ceil,fabs,exp
class SUBMINLIN(object):
    def __init__(self,data):
        self.data=data

    def InitDVC(self,sampleSize,n,q):
        self.sampleSize=sampleSize
        self.n=n
        self.cost=[0]*self.n
        self.costSum=0.0
        for i in range(self.n):
            tempElemetn=[i]
            tempElemetn.extend(self.data[i])
            tempValue=len(list(set(tempElemetn)))-q
            if tempValue>0:
                self.cost[i] =tempValue
            else:
                self.cost[i]=1
            self.costSum+=self.cost[i]
        self.gamma=1.0

    #calculate the indices of a solution
    def Position(self, s):
        return np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

    #calculate g(s)-c(s)
    def Objective_DVC(self,s):
        pos = self.Position(s)
        tempSet=[]
        #tempSet.extend(pos)
        for j in pos:
            tempSet.extend(self.data[j])
        tempSet.extend(pos)
        tempSet=list(set(tempSet))
        tempSum=len(tempSet)
        for item in pos:
            tempSum -= self.cost[item]
        return tempSum

    def Maginal_Gain_G_DVC(self,s,e):
        temp1=self.GS(s)

        if s[0,e]==0:
            s[0,e]=1
            temp2=self.GS(s)
            s[0,e]=0
            return temp2-temp1
        else:
            #temp1 = self.GS(s)
            return 0
        '''
        pos = self.Position(s)
        tempSet = []
        for j in pos:
            tempSet.extend(self.data[j])
        tempSet.extend(pos)
        tempSet = list(set(tempSet))
        tempSum = len(tempSet)
        tempSet.append(e)
        tempSet.extend(self.data[e])
        tempSet = list(set(tempSet))
        return len(tempSet)-tempSum
        '''





    def Distorted_Greedy(self,k):
        self.result = np.mat(np.zeros((1, self.n)), 'int8')
        self.constraint = k
        self.distortedGreedyVolume = [0] * self.constraint
        i=0
        selectedIndex=0
        while i< self.constraint:
            coefficent=pow(1-self.gamma*1.0/self.constraint,self.constraint-i-1)
            tempVolume=0
            for j in range(0,self.n):
                if self.result[0,j]==0:
                    margianGain_min_ce=coefficent*self.Maginal_Gain_G_DVC(self.result,j)-self.cost[j]
                    if margianGain_min_ce>tempVolume:
                        tempVolume=margianGain_min_ce
                        selectedIndex=j
            if tempVolume>0:
                self.result[0,selectedIndex]=1
            self.distortedGreedyVolume[i]=self.Objective_DVC(self.result)
            i+=1
        while i < self.constraint:
            self.distortedGreedyVolume[i] = self.distortedGreedyVolume[i-1]
            i += 1

    def StochasticDistortedGreedy(self,k):
        self.result = np.mat(np.zeros((1, self.n)), 'int8')
        self.constraint = k
        self.stochasticDistortedGreedyVolume = [0] * self.constraint
        selectedIndex=0
        i=0
        while i<self.constraint:
            sampeItems=sample(range(self.n),int(self.sampleSize))
            coefficent = pow(1 - self.gamma * 1.0 / self.constraint, self.constraint - i - 1)
            tempVolume = 0
            for j in sampeItems:
                if self.result[0,j]==0:
                    margianGain_min_ce=coefficent*self.Maginal_Gain_G_DVC(self.result,j)-self.cost[j]
                    if margianGain_min_ce>tempVolume:
                        tempVolume=margianGain_min_ce
                        selectedIndex=j
            if tempVolume>0:
                self.result[0,selectedIndex]=1
            self.stochasticDistortedGreedyVolume[i]=self.Objective_DVC(self.result)
            i+=1
        while i < self.constraint:
            self.stochasticDistortedGreedyVolume[i] = self.distortedGreedyVolume[i-1]
            i += 1

    def mutation(self, s):
        rand_rate = 1.0 / (self.n+self.constraint) # the dummy items are considered
        change = np.random.binomial(1, rand_rate, self.n+self.constraint)
        return np.abs(s - change)

    #calculate g(s)
    def GS(self,s):
        pos = self.Position(s)
        validePos=[]
        tempSet = []
        for j in pos:
            if j<self.n: #the dummy items are excluded when computing g(S)
                validePos.append(j)
                tempSet.extend(self.data[j])
        tempSet.extend(validePos)
        tempSet = list(set(tempSet))
        return len(tempSet)

    #calculate c(s)
    def CS(self,s):
        pos = self.Position(s)
        tempSum = 0.0
        for item in pos:
            if item < self.n:  # the dummy items are excluded when computing g(S)
                tempSum += self.cost[item]
        return tempSum

    def GESMO(self,k):
        self.constraint = k
        population = np.mat(np.zeros([1, self.n+k], 'int8'))  # initiate the population
        fitness = np.mat(np.zeros([1, 3]))
        popSize = 1
        t = 0  # the current iterate count
        iter1=0
        T = int(ceil(n*k * k * exp(1)))
        kn=int(self.constraint*self.n)
        while t<T:
            if iter1 == kn:
                iter1 = 0
                resultIndex = -1
                maxValue = float("-inf")
                for p in range(0, popSize):
                    if fitness[p, 1] <= self.constraint and fitness[p, 2] > maxValue:
                        maxValue = fitness[p, 2]
                        resultIndex = p
                print(fitness[resultIndex,:])
            iter1+=1
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
            offSpringFit = np.mat(np.zeros([1,3])) #comparable value, size, original value
            offSpringFit[0, 1] = offSpring[0, :].sum()
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] >= self.constraint+3:
                t += 1
                continue
            gs=self.GS(offSpring)
            cs=self.CS(offSpring)
            offSpringFit[0, 2]=gs-cs
            offSpringFit[0,0]=pow(1 - self.gamma / self.constraint, self.constraint - offSpringFit[0, 1])*gs-cs + self.costSum*offSpringFit[0,1]/self.constraint

            #update the population
            hasBetter = False
            for i in range(0, popSize):
                if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                        fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                    hasBetter = True
                    break
            if hasBetter == False:  # there is no better individual than offSpring
                Q = []
                for j in range(0, popSize):
                    if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                        continue
                    else:
                        Q.append(j)
                fitness = np.vstack((offSpringFit, fitness[Q, :]))  # update fitness
                population = np.vstack((offSpring, population[Q, :]))  # update population
            t = t + 1
            popSize = np.shape(fitness)[0]
        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, popSize):
            if fitness[p, 1] <= self.constraint and fitness[p, 2] > maxValue:
                maxValue = fitness[p, 2]
                resultIndex = p
        return fitness[resultIndex,2]


    def GSEMO_g_c(self, k):
        self.constraint = k
        popSize = 1
        t = 0  # the current iterate count
        population = np.mat(np.zeros([1, self.n+k], 'int8'))  # initiate the population
        fitness = np.mat(np.zeros([1, 2]))
        iter1 = 0
        #T=int(ceil((n+self.constraint)*k*k*exp(1)*exp(1)))
        T = int(ceil(self.n * k * k*exp(1.0)))
        kn = int(self.constraint * self.n)
        #startTime=time.time()
        while t < T:
            if iter1 == kn:
                iter1 = 0
                resultIndex = -1
                maxValue = float("-inf")
                for p in range(0, popSize):
                    if fitness[p, 1] <= self.constraint and fitness[p, 0] > maxValue:
                        maxValue = fitness[p, 0]
                        resultIndex = p
                print(fitness[resultIndex, :])
            iter1 += 1
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
            offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
            offSpringFit[0, 1] = offSpring[0, :].sum()
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] >= self.constraint + 3:
                t += 1
                continue
            gs = self.GS(offSpring)
            cs = self.CS(offSpring)
            offSpringFit[0, 0] = gs - cs
            hasBetter = False
            for i in range(0, popSize):
                if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                        fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                    hasBetter = True
                    break
            if hasBetter == False:  # there is no better individual than offSpring
                Q = []
                for j in range(0, popSize):
                    if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                        continue
                    else:
                        Q.append(j)
                fitness = np.vstack((offSpringFit, fitness[Q, :]))  # update fitness
                population = np.vstack((offSpring, population[Q, :]))  # update population
            t = t + 1
            popSize = np.shape(fitness)[0]
        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, popSize):
            if fitness[p, 1] <= self.constraint and fitness[p, 0] > maxValue:
                maxValue = fitness[p, 0]
                resultIndex = p
        return fitness[resultIndex, 0]


    def Multi_SDG(self, k):
        runtime=ceil(exp(1.0)*k*self.n/(self.sampleSize))
        maxValue=float('-inf')
        for i in range(runtime):
            self.StochasticDistortedGreedy(k)
            if self.stochasticDistortedGreedyVolume[k-1]>=maxValue:
                maxValue=self.stochasticDistortedGreedyVolume[k-1]
            #print(maxValue)
        return maxValue


def GetDVCData(fileName):
    node_neighbor=[]
    i=0
    file = open(fileName)
    lines = file.readlines()
    while i<945:
        currentLine=[]
        for line in lines:
            items=line.split()
            if int(items[0])-1 == int(i):
                currentLine.append(int(items[1])-1)
        node_neighbor.append(currentLine)
        i+=1
    file.close()
    return node_neighbor


if __name__=="__main__":

    #read data and normalize it
    data=GetDVCData('frb45-21-1.mis')
    n=945
    q=6
    constraint=100

    myObject.InitDVC(1,n,q)#sampleSize,n,q

    cardinality_lsit = []
    DGResult=[]
    SDGResult_mean=[]
    SDGResult_std=[]
    for i in range(1,11):
        cardinality_lsit.append(int(i*10))
    eps = 0.1
    for k in cardinality_lsit:
        tempResult=[]
        sampleSize = ceil(n * log(1.0 / eps) / k)
        if sampleSize>n:
            sampleSize=n
        myObject.InitDVC(sampleSize, n, q)  # sampleSize,n,q
        myObject.Distorted_Greedy(k)
        DGResult.append(myObject.distortedGreedyVolume[k-1])
        for i in range(20):
            myObject.StochasticDistortedGreedy(k)
            tempResult.append(myObject.stochasticDistortedGreedyVolume[k-1])
        SDGResult_mean.append(np.mean(tempResult))
        SDGResult_std.append(np.std(tempResult))
    print('Distorted Greedy')
    print(DGResult)
 
    print('Stochastic Distorted Greedy eps=0.1:')
    print(SDGResult_mean)
    print(SDGResult_std)

    #eps=0.2
    eps = 0.2
    SDGResult_mean = []
    SDGResult_std = []
    for k in cardinality_lsit:
        SDGTempResult = []
        sampleSize = ceil(n * log(1.0 / eps) / k)
        if sampleSize > n:
            sampleSize = n
        myObject.InitDVC(sampleSize, n, q)  # sampleSize,n,q
        for i in range(20):
            myObject.StochasticDistortedGreedy(k)
            SDGTempResult.append(myObject.stochasticDistortedGreedyVolume[k - 1])
        SDGResult_mean.append(np.mean(SDGTempResult))
        SDGResult_std.append(np.std(SDGTempResult))
    print('Stochastic Distorted Greedy eps=0.2:')
    print(SDGResult_mean)
    print(SDGResult_std)

    #GESMO
    GESMOResult_mean = []
    GESMOResult_std = []
    #cardinality_lsit=[20]
    for k in cardinality_lsit:
        GESMOTempResult=[]
        for i in range(20):
            GESMOTempResult.append(myObject.GESMO(k))
        GESMOResult_mean.append(np.mean(GESMOTempResult))
        GESMOResult_std.append(np.std(GESMOTempResult))
    print(GESMOResult_mean)
    print(GESMOResult_std)

    # GESMO_g_c
    g_c_Result_mean = []
    g_c_Result_std = []
    # cardinality_lsit=[20]
    for k in cardinality_lsit:
        tempResult = []
        for i in range(20):
            tempResult.append(myObject.GSEMO_g_c(k))
        g_c_Result_mean.append(np.mean(tempResult))
        g_c_Result_mean.append(np.std(tempResult))
    print(g_c_Result_mean)
    print(g_c_Result_mean)

    # Multi_SDG
    Multi_SDG_Result_mean = []
    Multi_SDG_Result_std = []
    # cardinality_lsit=[20]
    for k in cardinality_lsit:
        tempResult = []
        for i in range(20):
            eps = uniform(0.1, 0.5)
            sampleSize = ceil(n * log(1.0 / eps) / k)
            if sampleSize > n:
                sampleSize = n
            myObject.InitDVC(sampleSize, n, q)
            tempResult.append(myObject.Multi_SDG(k))
        Multi_SDG_Result_mean.append(np.mean(tempResult))
        Multi_SDG_Result_std.append(np.std(tempResult))
    print(Multi_SDG_Result_mean)
    print(Multi_SDG_Result_std)





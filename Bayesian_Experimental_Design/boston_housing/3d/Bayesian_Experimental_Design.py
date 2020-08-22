import sys
import numpy as np
from random import sample
from random import randint,random,uniform
from math import pow,log,ceil,sqrt,fabs,exp
class SUBMINLIN(object):
    def __init__(self,data):
        self.data=data
    def InitBO(self,alpha,sampleSize,sigma,reSetSigmaMatrix):
        self.sampleSize=sampleSize
        self.sigma=sigma
        [self.row,self.n]=np.shape(self.data)
        self.costSum=0.0
        if reSetSigmaMatrix:
            '''
            A =np.random.randn(self.row,self.row)# standard normal distribution
            outFile=open('randm_A_matrix_7d.txt','w')
            for i in range(self.row):
                for j in range(self.row):
                    outFile.write(str(A[i,j])+' ')
                outFile.write('\n')
            outFile.close()
            '''
            #read the random matrix A
            A=np.mat(np.zeros((self.row,self.row)))
            outFile=open('./data/randm_A_matrix_3d.txt')
            lines=outFile.readlines()
            for i in range(self.row):
                items=lines[i].split()
                for j in range(self.row):
                    A[i,j]=float(items[j])
            outFile.close()
            D = np.diag([pow(i*1.0/self.row, 2) for i in range(1,self.row+1)])
            self.Sigma=np.dot(np.dot(A,D),np.transpose(A))
        self.cost=[0]*self.n
        self.tempTraceSum=np.trace(self.Sigma)
        self.tempInverse=np.linalg.inv(self.Sigma)

        #sum up the cost
        for i in range(self.n):
            self.cost[i]=alpha*(self.tempTraceSum-np.trace(np.linalg.inv(self.tempInverse+pow(1.0/sigma,2)*np.dot(self.data[:,i],np.transpose(self.data[:,i])))))
            self.costSum += self.cost[i]

        #calculate the maximum eigen value of matrix Sigma
        [eigenValue,featureVector]=np.linalg.eig(self.Sigma)
        maxEigenValue=np.max(eigenValue)
        lineSuqre=[pow(np.linalg.norm(self.data[:,i]),2) for i in range(self.n)]
        s_squre=np.max(lineSuqre)
        self.gamma=1.0/(1+s_squre*maxEigenValue*pow(1.0/sigma,2))
        if reSetSigmaMatrix:

            print('the lower bound of gamma is: %f'%(self.gamma))

    # calculate the indices of a solution s
    def Position(self, s):
        return np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

    #calculate g(s)-c(s)
    def Objective_BO(self,s):
        pos = self.Position(s)
        tempSum=self.tempTraceSum-np.trace(np.linalg.inv(self.tempInverse+pow(1.0/self.sigma,2)*np.dot(self.data[:,pos],np.transpose(self.data[:,pos]))))
        for item in pos:
            tempSum -= self.cost[item]
        return tempSum

    #calculate g(s)
    def Maginal_Gain_G_BO1(self, s):
        pos = self.Position(s)
        tempSum = self.tempTraceSum - np.trace(np.linalg.inv(self.tempInverse + pow(1.0 / self.sigma, 2) * np.dot(self.data[:, pos], np.transpose(self.data[:, pos]))))
        return tempSum

    def Distorted_Greedy(self,k):
        self.result = np.mat(np.zeros((1, self.n)), 'int8')
        self.constraint = k
        self.distortedGreedyVolume = [0] * self.constraint
        i=0
        selectedIndex=0
        while i< self.constraint:
            coefficent=pow(1-self.gamma/self.constraint,self.constraint-i-1)
            tempVolume=0
            for j in range(0,self.n):
                if self.result[0,j]==0:
                    self.result[0, j] = 1
                    margianGain_min_ce1 = self.Maginal_Gain_G_BO1(self.result)
                    self.result[0, j] = 0
                    margianGain_min_ce2 = self.Maginal_Gain_G_BO1(self.result)
                    margianGain_min_ce = coefficent * (margianGain_min_ce1 - margianGain_min_ce2) - self.cost[j]
                    if margianGain_min_ce>tempVolume:
                        tempVolume=margianGain_min_ce
                        selectedIndex=j
            if tempVolume>0:
                self.result[0,selectedIndex]=1
            self.distortedGreedyVolume[i]=self.Objective_BO(self.result)
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
            coefficent = pow(1 - self.gamma / self.constraint, self.constraint - i - 1)
            tempVolume = 0
            for j in sampeItems:
                if self.result[0,j]==0:
                    self.result[0,j]=1
                    margianGain_min_ce1=self.Maginal_Gain_G_BO1(self.result)
                    self.result[0, j] = 0
                    margianGain_min_ce2 =self.Maginal_Gain_G_BO1(self.result)
                    margianGain_min_ce=coefficent*(margianGain_min_ce1-margianGain_min_ce2)-self.cost[j]
                    if margianGain_min_ce>tempVolume:
                        tempVolume=margianGain_min_ce
                        selectedIndex=j
            if tempVolume>0:
                self.result[0,selectedIndex]=1
            self.stochasticDistortedGreedyVolume[i]=self.Objective_BO(self.result)
            i+=1
        while i < self.constraint:
            self.stochasticDistortedGreedyVolume[i] = self.distortedGreedyVolume[i-1]
            i += 1

    def mutation(self, s):
        rand_rate = 1.0 / (self.n+self.constraint) # the dummy items are considered
        change = np.random.binomial(1, rand_rate, self.n+self.constraint)
        return np.abs(s - change)

    #calculate g(s)
    def GS(self, s):
        tempPos = self.Position(s)
        pos=[]
        for item in tempPos:
            if item<self.n:
                pos.append(item)
        tempSum = self.tempTraceSum - np.trace(np.linalg.inv(
            self.tempInverse + pow(1.0 / self.sigma, 2) * np.dot(self.data[:, pos], np.transpose(self.data[:, pos]))))
        return tempSum

    # calculate c(s)
    def CS(self, s):
        tempPos = self.Position(s)
        pos=[]
        for item in tempPos:
            if item < self.n:
                pos.append(item)
        tempSum = 0.0
        for item in pos:
            tempSum += self.cost[item]
        return tempSum

    def GSEMO(self, k):
        self.constraint = k
        population = np.mat(np.zeros([1, self.n+k], 'int8'))  # initiate the population
        fitness = np.mat(np.zeros([1, 3]))
        popSize = 1
        t = 0  # the current iterate count
        iter1 = 0
        #T=int(ceil((n+self.constraint)*k*k*exp(1)*exp(1)))
        T = int(ceil(n * k * k*exp(1.0)))
        kn = int(self.constraint * self.n)
        while t < T:
            if iter1 == kn:
                iter1 = 0
                resultIndex = -1
                maxValue = float("-inf")
                for p in range(0, popSize):
                    if fitness[p, 1] <= self.constraint and fitness[p, 2] > maxValue:
                        maxValue = fitness[p, 2]
                        resultIndex = p
                print(fitness[resultIndex, :])
            iter1 += 1
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
            offSpringFit = np.mat(np.zeros([1, 3]))  # comparable value, size, original value
            offSpringFit[0, 1] = offSpring[0, :].sum()
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] >= self.constraint + 3:
                t += 1
                continue
            gs = self.GS(offSpring)
            cs = self.CS(offSpring)
            offSpringFit[0, 2] = gs - cs
            offSpringFit[0, 0] = pow(1 - self.gamma / self.constraint,self.constraint - offSpringFit[0, 1]) * gs - cs + self.costSum * offSpringFit[0, 1] / self.constraint

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
        return fitness[resultIndex, 2]

    def GSEMO_g_c(self, k):
        self.constraint = k
        population = np.mat(np.zeros([1, self.n], 'int8'))  # initiate the population
        fitness = np.mat(np.zeros([1, 2]))
        popSize = 1
        t = 0  # the current iterate count
        iter1 = 0
        # T=int(ceil((n+self.constraint)*k*k*exp(1)*exp(1)))
        T = int(ceil(n * k * k * exp(1.0)))
        kn = int(self.constraint * self.n)
        # startTime=time.time()
        while t < T:
            if iter1 == kn:
                iter1 = 0
                resultIndex = -1
                maxValue = float("-inf")
                for p in range(0, popSize):
                    if fitness[p, 1] <= self.constraint and fitness[p, 0] > maxValue:
                        maxValue = fitness[p, 0]
                        resultIndex = p
                # print(fitness[resultIndex, :])
                # if maxValue>=self.distortedGreedyVolume[k-1]:
                # endTime=time.time()
                # print('GSEMO_g-c cost time is :%.8s s'%(endTime-startTime))
            iter1 += 1
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
            offSpring = self.mutation_no_dummy(s)  # every bit will be flipped with probability 1/n
            offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
            offSpringFit[0, 1] = offSpring[0, :].sum()
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] >= self.constraint + 3:
                t += 1
                continue
            gs = self.GS(offSpring)
            cs = self.CS(offSpring)
            offSpringFit[0, 0] = gs - cs
            # offSpringFit[0, 0] = pow(1 - self.gamma / self.constraint,self.constraint - offSpringFit[0, 1]) * gs - cs + self.costSum * offSpringFit[0, 1] / self.constraint

            # update the population
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
        runtime = ceil(exp(1.0) * k * self.n / self.sampleSize)
        maxValue = float('-inf')
        for i in range(runtime):
            self.StochasticDistortedGreedy(k)
            if self.stochasticDistortedGreedyVolume[k - 1] >= maxValue:
                maxValue = self.stochasticDistortedGreedyVolume[k - 1]
        return maxValue


def GetBOData(fileName):
    file=open(fileName)
    lines=file.readlines()
    data=[]
    for line in lines:
        items=line.split()
        newLine=[]
        for item in items:
            newLine.append(float(item))
        data.append(newLine)
    data=np.mat(data)
    file.close()
    return data


if __name__=="__main__":

    #read data and normalize it
    data=GetBOData('./data/Boston_Housing.txt')
    matMean = data.mean(0)
    matStd = data.std(0)
    norm_colum = matStd.nonzero()[1]
    data[:,norm_colum] = (data[:,norm_colum] - matMean[:,norm_colum]) / matStd[:,norm_colum]
    data = np.transpose(data)
    [m,n]=np.shape(data)

    myObject=SUBMINLIN(data)
    alpha=0.8
    sigma=m*3.0#1.0/sqrt(m)
    constraint=20
    mode=1# 1,2,3 are standard ,orignla(gs-cs),no rear
    myObject.InitBO(alpha, 1, sigma, True)  # alpha,sampleSize,sigma,reSetSigmaMatrix

    cardinality_lsit = range(5, 21)
    myObject.Distorted_Greedy(constraint)
    #eps=0.1
    eps = 0.1
    DGResult=[]
    SDGResult_mean = []
    SDGResult_std = []
    for k in cardinality_lsit:
        tempResult = []
        sampleSize = ceil(n * log(1.0 / eps) / k)
        if sampleSize > n:
            sampleSize = n
        myObject.InitBO(alpha, sampleSize, sigma, False)
        myObject.Distorted_Greedy(k)
        DGResult.append(myObject.distortedGreedyVolume[k-1])
        for i in range(20):
            myObject.StochasticDistortedGreedy(k)
            tempResult.append(myObject.stochasticDistortedGreedyVolume[k - 1])
        SDGResult_mean.append(np.mean(tempResult))
        SDGResult_std.append(np.std(tempResult))
    print('Distorted Greedy')
    print(DGResult)
    print('Stochastic Distorted Greedy eps=0.1:')
    print(SDGResult_mean)
    print(SDGResult_std)

    #eps=0.2
    eps=0.2
    SDGResult_mean = []
    SDGResult_std = []
    for k in cardinality_lsit:
        tempResult = []
        sampleSize = ceil(n * log(1.0 / eps) / k)
        if sampleSize > n:
            sampleSize = n
        myObject.InitBO(alpha, sampleSize, sigma, False)
        for i in range(20):
            myObject.StochasticDistortedGreedy(k)
            tempResult.append(myObject.stochasticDistortedGreedyVolume[k - 1])
        SDGResult_mean.append(np.mean(tempResult))
        SDGResult_std.append(np.std(tempResult))
    print('Stochastic Distorted Greedy eps=0.2:')
    print(SDGResult_mean)
    print(SDGResult_std)


    print('GESMO')
    GESMOResult1_mean=[]
    GESMOResult1_std = []
    for k in cardinality_lsit:
        GESMOTempResult1 = []
        for i in range(20):
            print('GESMO k=%d i=%d' % (k, i))
            GESMOTempResult1.append(myObject.GSEMO(k))
        GESMOResult1_mean.append(np.mean(GESMOTempResult1))
        GESMOResult1_std.append(np.std(GESMOTempResult1))
    print(GESMOResult1_mean)
    print(GESMOResult1_std)

    print('GESMO_g_c')
    g_c_Result1_mean = []
    g_c_Result1_std = []
    for k in cardinality_lsit:
        tempResult = []
        for i in range(20):
            print('GESMO_g_c k=%d i=%d' % (k, i))
            tempResult.append(myObject.GSEMO_g_c(k))
        g_c_Result1_mean.append(np.mean(tempResult))
        g_c_Result1_std.append(np.std(tempResult))
    print(g_c_Result1_mean)
    print(g_c_Result1_std)

    print('Mulit_SDG')
    Multi_SDG_Result1_mean = []
    Multi_SDG_Result1_std = []
    for k in cardinality_lsit:
        tempResult = []
        for i in range(20):
            eps = uniform(0.1, 0.5)
            sampleSize = ceil(n * log(1.0 / eps) / k)
            if sampleSize > n:
                sampleSize = n
            myObject.InitBO(alpha, sampleSize, sigma, False)
            print('Mulit_SDG k=%d i=%d' % (k, i))
            tempResult.append(myObject.Multi_SDG(k))
        Multi_SDG_Result1_mean.append(np.mean(tempResult))
        Multi_SDG_Result1_std.append(np.std(tempResult))
    print(Multi_SDG_Result1_mean)
    print(Multi_SDG_Result1_std)






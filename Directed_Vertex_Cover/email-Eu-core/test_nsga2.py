'''
Created on 10/01/2011

@author: 04610922479
'''
import random, math
from nsga2 import Solution
from nsga2 import NSGAII
import matplotlib.pyplot as plt
import numpy as np
#from __builtin__ import max

#from nsga2.nsga2 import Solution
class CalculateObj(object):
    def __init__(self,data,sampleSize,constraint,n,q):
        self.data=data
        self.sampleSize = sampleSize
        self.constraint=constraint
        self.n = n
        self.cost = [0] * self.n
        self.costSum = 0.0
        for i in range(self.n):
            tempElemetn = [i]
            tempElemetn.extend(self.data[i])
            tempValue = len(list(set(tempElemetn))) - q
            if tempValue > 0:
                self.cost[i] = tempValue
            else:
                self.cost[i] = 1
            self.costSum += self.cost[i]
        self.gamma = 1.0

    # calculate the indices of a solution s
    def Position(self, s):
        return np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

    def GS(self, s):
        pos = self.Position(s)
        validePos = []
        tempSet = []
        for j in pos:
            if j < self.n:  # the dummy items are excluded when computing g(S)
                validePos.append(j)
                tempSet.extend(self.data[j])
        tempSet.extend(validePos)
        tempSet = list(set(tempSet))
        return len(tempSet)

    def CS(self, s):
        pos = self.Position(s)
        tempSum = 0.0
        for item in pos:
            if item < self.n:  # the dummy items are excluded when computing g(S)
                tempSum += self.cost[item]
        return tempSum
    def Evaluate(self,s):
        size_s=s.sum()
        #print(size_s,self.constraint)
        o1=44444444444
        if size_s<=self.constraint+2:
            gs=self.GS(s)
            cs=self.CS(s)
            #o1=gs-cs
            #if abs(gs-cs)>0.000001:
                #print(gs-cs)
            o1=44444444444-(pow(1 - self.gamma / self.constraint,self.constraint - size_s) * gs - cs + self.costSum * size_s / self.constraint)
            #o1=10000000.0-pow(1 - self.gamma / self.constraint,self.constraint - size_s) * gs - cs + self.costSum * size_s / self.constraint
        else:
            size_s=44444444444
        #print(o1,size_s)
        return o1,size_s
class T1Solution(Solution):
    '''
    Solution for the T1 function.
    '''
    def __init__(self,s,calculateObj):
        
        '''
        Constructor.
        '''
        Solution.__init__(self, 2)#two objectives 
        self.s=s
        self.n=np.shape(s)[1]
        self.calculateObj=calculateObj
        #self.xmin = 0.0
        #self.xmax = 1.0
        
        #for _ in range(30):
            #self.attributes.append(random.random())
        
        self.evaluate_solution()
        
    def evaluate_solution(self):
        '''
        Implementation of method evaluate_solution() for T1 function.
        '''
        temp1,temp2=self.calculateObj.Evaluate(self.s)
        self.objectives[0] = temp1
        self.objectives[1] = temp2
        #print(self.objectives)
        
    def crossover(self, other):
        '''
        Crossover of T1 solutions.
        '''
        randVec=np.random.rand(1,self.n)
        child_solution = T1Solution(np.mat(np.zeros((1,self.n)),'int8'),self.calculateObj)
        index1=randVec[0,:]<0.5
        index2=randVec[0,:]>=0.5
        child_solution.s[0,index1]=self.s[0,index1]
        child_solution.s[0,index2]=other.s[0,index2]
        #for i in range(30):
            #child_solution.attributes[i] = math.sqrt(self.attributes[i] * other.attributes[i])
        return child_solution
    
    def mutate(self):
        '''
        Mutation of T1 solution.
        '''
        #self.attributes[random.randint(0, 29)] = random.random())
        change = np.random.binomial(1, 1.0 / self.n, self.n)
        self.s=np.abs(self.s - change)

def GetDVCData(fileName):
    node_neighbor=[]
    i=0
    file = open(fileName)
    lines = file.readlines()
    while i<1005:
        currentLine=[]
        for line in lines:
            items=line.split()
            if int(items[0]) == int(i):
                currentLine.append(int(items[1]))
        node_neighbor.append(currentLine)
        i+=1
    file.close()
    return node_neighbor


if __name__=="__main__":

    #read data and normalize it
    data=GetDVCData('email-Eu-core.txt')

    myObject=SUBMINLIN(data)
    n=1005
    q=6
    sampleSize=1
    mean=[]
    std=[]
    constraint_list=[10,20,30,40,50,60,70,80,90,100]
    for constraint in constraint_list:
        calculateObj=CalculateObj(data,sampleSize,constraint,n,q)
        nsga2 = NSGAII(2, 0.1, 1.0)
        popSize=100
        iterationNum=math.ceil(math.exp(1)*constraint*constraint*n/popSize)
        tempResult=[]
        P=[]
        for i in range(20):#run 20 times
            del P[:]
            #initialize the population
            solutionSize = np.random.randint(0,constraint+1,size=popSize)
            for item in solutionSize:
                index=np.random.randint(0,n+constraint,size=item)# posiztion index
                s=np.mat(np.zeros((1,n+constraint)),'int8')
                s[0,index]=1
                P.append(T1Solution(s,calculateObj))
            
            nsga2.run(P, popSize, iterationNum)
            maxValue=float('-inf')
            solution=None
            for item in P:
                if item.s.sum()<=constraint:
                    tempValue=calculateObj.GS(item.s)-calculateObj.CS(item.s)
                    if maxValue<tempValue:
                        maxValue=tempValue
            tempResult.append(maxValue)
        temp=np.mat(tempResult)
        print(temp)
        mean.append(np.mean(temp,axis=1))
        std.append(np.std(temp,axis=1))
    print(mean)
    print(std)

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
    def __init__(self,data,alpha,sigma,reSetSigmaMatrix,k):
        self.data=data
        self.sigma=sigma
        self.constraint=k
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
            outFile=open('./data/randm_A_matrix_11d.txt')
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
    
    def Evaluate(self,s):
        size_s=s.sum()
        #print(size_s,self.constraint)
        o1=444444444
        if size_s<=self.constraint+2:
            gs=self.GS(s)
            cs=self.CS(s)
            #o1=gs-cs
            #if abs(gs-cs)>0.000001:
                #print(gs-cs)
            o1=1000-(pow(1 - self.gamma / self.constraint,self.constraint - size_s) * gs - cs + self.costSum * size_s / self.constraint)
            #o1=10000000.0-pow(1 - self.gamma / self.constraint,self.constraint - size_s) * gs - cs + self.costSum * size_s / self.constraint
        else:
            size_s=444444444
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
        #print(np.shape(self.s),np.shape(other.s))
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

    
if __name__ == '__main__':
    
    #read data and initialize calculate objective 
    data=GetBOData('./data/segment_data.txt')
    matMean = data.mean(0)
    matStd = data.std(0)
    norm_colum = matStd.nonzero()[1]
    data[:,norm_colum] = (data[:,norm_colum] - matMean[:,norm_colum]) / matStd[:,norm_colum]
    data = np.transpose(data)
    [m,n]=np.shape(data)

    alpha=0.8

    sigma=m*11.0#1.0/sqrt(m)
    reSetSigmaMatrix=True
    mean=[]
    std=[]
    cardinalities=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    for constraint in cardinalities:
        calculateObj=CalculateObj(data,alpha,sigma,reSetSigmaMatrix,constraint)
        nsga2 = NSGAII(2, 0.1, 1.0)
        popSize=100
        iterationNum=math.ceil(math.exp(1)*constraint*constraint*(n+constraint)/popSize)
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
        print('constraint is %f '%(constraint))
        print(temp)
        mean.append(np.mean(temp,axis=1))
        std.append(np.std(temp,axis=1))
    print(mean)
    print(std)

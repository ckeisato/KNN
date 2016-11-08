import sys
import numpy as np
import math


class Perceptron:
    trainData = []
    testData = []
    weights = []
    numAttributes = []
    learning = 0.0
    top = [] 

    def __init__(self, trainfile, testfile, learning):
        self.trainData = []
        self.testData = []
        self.weights = []
        self.learning = float(learning)
        
        #filling out the training matrix
        trainRead = open(trainfile, 'r')
        for row in trainRead:
            if row == "@data\n":
                break

        for row in trainRead:
            row = row.strip()
            row = row.split(',')
            row = map(float, row)
            self.trainData.append(row)

        #filling out the test matrix
        testRead = open(testfile, 'r')
        for row in testRead:
            self.top.append(row)
            if row == "@data\n":
                break
        for row in testRead:
            row = row.strip()
            row = row.split(',')
            row = map(float, row)
            self.testData.append(row)

        self.numAttributes = len(self.trainData[0]) - 1


    def Normalize(self):
        n = len(self.trainData)
        stanDev = []
        collectArray = []
        featMean = []
        for i in range(0,self.numAttributes):
            for j in range(0,n):
                collectArray.append(self.trainData[j][i])
            stanDev.append(np.std(np.array(collectArray), ddof=1))
            featMean.append(np.mean(np.array(collectArray)))
            collectArray = []
            
        for i in range(0,n):
            for j in range(0,self.numAttributes):
                actual = self.trainData[i][j]
                value = (actual-featMean[j])/stanDev[j]
                self.trainData[i][j] = value

        n = len(self.testData)
        collectArray = []
        for i in range(0,n):
            for j in range(0,self.numAttributes):
                actual = self.testData[i][j]
                value = (actual - featMean[j])/stanDev[j]
                self.testData[i][j] = value 
                     
    
    def StartPerceptron(self):
         
        for i in range(0, self.numAttributes+1):
            self.weights.append(0.0)
            
        for i in range(0, 500):
            for row in self.trainData:
                target = row[self.numAttributes]
                row.insert(0,1.0)
                dotProduct = np.dot(np.array(row[0:self.numAttributes+1]),\
                np.array(self.weights))
                diff = target-dotProduct
                for i in range(0, (self.numAttributes+1)):
                    change = self.learning*diff*row[i]
                    self.weights[i] = self.weights[i] + change
                       
        c = 0
        for row in self.testData:
            actual = row.pop()
            row.insert(0,1)
            dotProduct = np.dot(np.array(row), np.array(self.weights))
            
            if(dotProduct >= 0):
                if (actual != 1):
                    print(str(c)+' actual '+str(actual)+' '+str(dotProduct)) 
                    c+=1 
                row.append(1)
            if(dotProduct < 0):
                if (actual != -1):
                    print(str(c)+' actual '+str(actual)+' '+str(dotProduct))
                    c+=1 
                row.append(-1)
            
         

#java perceptron -eta 0.1 train.arff test.arff output.arff
perceptron = Perceptron(sys.argv[3], sys.argv[4], sys.argv[2])
perceptron.Normalize()
perceptron.StartPerceptron()

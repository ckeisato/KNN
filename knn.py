import arff
import math
import sys
import numpy as np


class KNN:
    trainData = []
    testData = []
    numAttributes = int
    class1 = []
    classNeg1 = []
    top = []
    output = str
    K = int

    def __init__(self, trainfile, testfile):
        self.trainData = []
        self.testData = []
        
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

    #get majority class
    def chooseClass(self, smallest):
        one = 0
        negOne = 0
        for row in smallest:
            if row[1] == 1.0:
                one += 1
            if row [1] == -1.0:
                negOne += 1
        if one > negOne:
            return 1
        if negOne > one:
            return -1

    def getDistance(self, train, test):
        sumofSq = 0
        for i in range(0,self.numAttributes):
            sumofSq += (test[i] - train[i])**2
        squareRT = math.sqrt(sumofSq)
        return squareRT
            

    # returns -1 or 1 class as determined by comparison with training data
    def getKNN(self, testRow):
        increment = self.K
        smallestDist = []

        for trainrow in self.trainData:
            dist = self.getDistance(trainrow, testRow)
            classify = trainrow[len(trainrow)-1]

            if len(smallestDist) < self.K:
                smallestDist.append([dist, classify])
            else:
                biggest = max(smallestDist)
                if dist < biggest:
                    biggest[0] = dist
                    biggest[1] = classify
        out = self.chooseClass(smallestDist)
        return out

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

 
    def startKNN(self, K, output):
        self.K = float(K)
        self.output = output

       # count = 0
        for row in self.testData:
            old = row.pop()
            new = self.getKNN(row)
        #    if old != new:
         #       count+=1
          #      print(str(count) + " old " + str(old) + " new " + str(new))
            row.append(new)
        
        output = open(self.output, 'w+')
        for row in self.top:
            output.write(str(row))
        for row in self.testData:
            last = row.pop()
            for item in row:
                output.write(str(item) + ',')
            output.write(str(last)+'\n')



# java kNN -Z 1 -k 5 train.arff test.arff output.arff

KNNObject = KNN(sys.argv[5], sys.argv[6])
if sys.argv[2] == '1':
    KNNObject.Normalize()
KNNObject.startKNN(sys.argv[4], sys.argv[7])

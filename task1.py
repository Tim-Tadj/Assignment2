import sys
import os
import random
import math
import matplotlib.pyplot as plt
import time

class dataset():
    def __init__(self, descriptors):
        self.flowers = []
        self.descriptors = descriptors
        self.testing = []
        self.training = []
    def addFlower(self, flower):
        flowerDict = {}
        for i in range(len(self.descriptors)):
            if i < 4:
                flowerDict[self.descriptors[i]] = float(flower[i])
            else:
                flowerDict[self.descriptors[i]] = flower[i]
        self.flowers.append(flowerDict)
    

    def spilt_data(self, percent):
        if not (0 <= percent <= 1):
            print("invalid input to spilt data")
            return
        self.training = random.sample(self.flowers, k = int((1-percent) * len(self.flowers)) )

        self.testing = []
        for i in self.flowers:
            if i not in self.training:
                self.testing.append(i)

    def euclidean_distance(self, flower1, flower2):
        euclideanSum = 0
        for feature, value in flower1.items():
            if not isinstance(value, str):
                euclideanSum += (value - flower2[feature])**2
        return math.sqrt(euclideanSum)

    #takes a training set, a single flower and an integer k, 
    #returns the k nearest neighbours of the flower in the training set
    def kNN(self, flower, k):
        distanceToFlower = []
        for trainingFlower in self.training:
            
            distance = self.euclidean_distance(trainingFlower, flower)

            #if no flowers exist
            if len(distanceToFlower) == 0:
                distanceToFlower.append([distance, trainingFlower])

            #if set is full
            elif len(distanceToFlower) >= k and distance < distanceToFlower[-1][0]:
                for i in range(len(distanceToFlower)):
                    if distance < distanceToFlower[i][0]:
                        distanceToFlower.insert(i, [distance, trainingFlower])
                        distanceToFlower.pop()
                        break
                    
            else:
                for i in range(len(distanceToFlower)):
                    if distance < distanceToFlower[i][0]:
                        distanceToFlower.insert(i, [distance, trainingFlower])
        
        return [b[1] for b in distanceToFlower] #return a list of k nearest flowers

    def flowerType(self, flower, kNearest, ):
        flowerCount = {}
        for nflower in kNearest:
            if nflower["species"] not in flowerCount:
                flowerCount[nflower["species"]] = 1/ self.euclidean_distance(flower, nflower)
            else:
                flowerCount[nflower["species"]] += 1/ self.euclidean_distance(flower, nflower)

        result = ""
        prev_count = 0
        for species, count in flowerCount.items():
            if count > prev_count:
                result = species
        
        return result

    def accuracy(self, k):
        accuracys = []
        for flower in self.testing:
            if self.flowerType(flower, self.kNN(flower, k)) == flower["species"]:
                accuracys.append(1)
            else:
                accuracys.append(0)
        # print(sum(accuracys), len(accuracys))
        return ( sum(accuracys) / len(accuracys) ) * 100

        

start = time.time()
with open(os.path.join(sys.path[0], "iris.csv"), "r") as openf:
    params = openf.readline().rstrip().split(",")
    mydata = dataset(params)
    for line in openf:
        mydata.addFlower(line.rstrip().split(","))
    mydata.spilt_data(0.3)
    kvalues = []
    kaccuracys = []
    for k in range(1, len(mydata.training)):
        kvalues.append(k)
        kaccuracys.append(mydata.accuracy(k))
    finish = time.time()
    print("done in :", finish-start, "s")
    # for i in range(len(kvalues)):
    #     print(kvalues[i], kaccuracys[i])
    plt.plot(kvalues, kaccuracys)
    plt.show()
        
        


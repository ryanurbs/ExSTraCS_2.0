"""
Name:        SURF.py
Authors:     Gediminas Bertasius and Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     December 4, 2013
Modified:    August 25,2014
Description: Surf algorithm computes the score of each attribute evaluating their strength based on nearest neighbours.
             Returns a list of attribute scores.
             
---------------------------------------------------------------------------------------------------------------------------------------------------------
ReBATE V1.0: includes stand-alone Python code to run any of the included/available Relief-Based algorithms designed for attribute filtering/ranking.
These algorithms are a quick way to identify attributes in the dataset that may be most important to predicting some phenotypic endpoint.  These scripts output
an ordered set of attribute names, along with respective scores (uniquely determined by the particular algorithm selected).  Certain algorithms require key
run parameters to be specified.  This code is largely based on the Relief-Based algorithms implemented in the Multifactor Dimensionality Reduction (MDR) software.
However these implementations have been expanded to accomodate continuous attributes (and continuous attributes mixed with discrete attributes) as well as a 
continuous endpoint.  This code also accomodates missing data points.  Built into this code, is a strategy to automatically detect from the loaded data, these 
relevant characteristics.

Copyright (C) 2013 Ryan Urbanowicz 
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, 
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def Run_SURF(data, reliefSampleFraction):
    """  Called to run the SURF algorithm.  
    #PARAM x- is a matrix containing the attributes of all instances in the dataset 
    #PARAM y- is a matrix containing the class of a data instance """
    
    x = [ row[0] for row in data.trainFormatted ] 
    y = [ row[1] for row in data.trainFormatted ] 
    
    print("Running SURF Algorithm...")
    maxInst=int(float(reliefSampleFraction)*len(x)) #param m-number of iteration to run during the ReliefF procedure
    scores=SURF(x,y,maxInst,data,reliefSampleFraction)
    print("SURF run complete.")
    
    return scores


def SURF(x,y,maxInst,data,reliefSampleFraction):
    """ Controls major SURF loops. """
    ScoreList=[]
    for i in range(data.numAttributes): #initializing attributes' scores to 0
        ScoreList.append(0)
    
    #Precompute distances between all unique instance pairs within the dataset.
    print("Precomputing Distance Array")
    distanceObject = calculateDistanceArray(x,data,maxInst)
    distanceArray = distanceObject[0]
    averageDistance = distanceObject[1]
    print("Computed")
    
    #For MulitClass Array Only
    multiclass_map = None
    if data.discretePhenotype and len(data.phenotypeList) > 2:
        multiclass_map = makeMultiClassMap(y,maxInst,data)
        
    for inst in range(maxInst): #evaluating each attribute over runIter runs 
        NN=find_nearest_neighbours_SURF(averageDistance,inst,distanceArray,maxInst) #and finding its nearest neighbours
        if len(NN)>0:
            for j in range(data.numAttributes):
                ScoreList[j]+=evaluate_SURF(x,y,NN,j,inst,data,multiclass_map,maxInst)

    return ScoreList
  
        
def calculateDistanceArray(x,data,maxInst):
    """ In SURF this method precomputes both the distance array and the average distance """
    #make empty distance array container (we will only fill up the non redundant half of the array
    distArray = []
    aveDist = 0
    count = 0
    for i in range(maxInst):
        distArray.append([])
        for j in range(maxInst):
            distArray[i].append(None)
            
    for i in range(1, maxInst):
        for j in range(0,i):
            distArray[i][j] = calculate_distance(x[i],x[j],data)
            count += 1
            aveDist += distArray[i][j]
            
    aveDist = aveDist/float(count)
    returnObject = [distArray,aveDist]
    return returnObject
        

def makeMultiClassMap(y, maxInst, data):
    #finding number of classes in the dataset and storing them into the map
    multiclass_map={}
    
    for i in range(maxInst):
        if (y[i] not in multiclass_map):
            multiclass_map[y[i]]=0
        else:
            multiclass_map[y[i]]+=1
            
    for each in data.phenotypeList: #For each class store probability of class occurrence in dataset.
        multiclass_map[each] = multiclass_map[each]/float(maxInst)
      
    return multiclass_map
    
  

def find_nearest_neighbours_SURF(averageDistance,inst,distanceArray,maxInst):
    """ Method that finds nearest neighbours of the entire dataset based either on distance metric or specification of k nearest neighbours 
    #PARAM x- matrix containing the attributes of all of the data instances
    #PARAM y- matrix containing the class labels of all the data instances
    #PARAM k- some integer number denoting number of nearest neighbours to consider
    #PARAM r-None if user wants nearest neighbours of all data instance
    #      or index of a data instance which the user wants to consider  """  
    NN=[]
    min_indices=[] 

    for j in range(maxInst):
        if inst != j:
            locator = [inst,j]
            locator = sorted(locator, reverse=True) #Access corect half of table (result of removed table redundancy)
            d = distanceArray[locator[0]][locator[1]]
            if d<averageDistance:
                min_indices.append(j)
            
    for j in range(len(min_indices)):
        NN.append(min_indices[j])
    
    return NN


def evaluate_SURF(x,y,NN,feature,inst,data,multiclass_map,maxInst):  
    """ Method evaluating the score of an attribute
    #PARAM x-matrix with the attributes of all dataset instances
    #PARAM y-matrix with the class labels of all dataset instances
    #PARAM NN-nearest neighbour matrix for each instance in the dataset
    #PARAM r-an index of a randomly selected data instance
    #PARAM feature-an attribute that should be evaluated """
    diff = 0
    if not data.discretePhenotype: #if continuous phenotype
        same_class_bound=data.phenSD #boundary to determine similarity between classes for continuous attributes
        
    if data.attributeInfo[feature][0]: #Continuous Attribute
        #determining boundaries for continuous attributes
        min_bound=data.attributeInfo[feature][1][0]
        max_bound=data.attributeInfo[feature][1][1]
    
    diff_hit=0 #initializing the score to 0
    diff_miss=0
    
    count_hit=0
    count_miss=0
    
    if data.discretePhenotype:
        if len(data.phenotypeList) > 2: #multiclass endpoint
            class_Store = {}
            missClassPSum = 0
            for each in multiclass_map:
                if each != y[inst]: #Store all miss classes
                    class_Store[each] = [0,0] #stores cout_miss and diff_miss
                    missClassPSum += multiclass_map[each]
            
            for i in range(len(NN)):  #for all nearest neighbors
                if x[inst][feature]!=data.labelMissingData and x[NN[i]][feature]!=data.labelMissingData: # add appropriate normalization.
                    if y[inst]==y[NN[i]]: #HIT
                        count_hit+=1
                        if x[inst][feature]!=x[NN[i]][feature]:
                            if data.attributeInfo[feature][0]: #Continuous Attribute
                                diff_hit-=abs(x[inst][feature]-x[NN[i]][feature])/(max_bound-min_bound)
                            else:#Discrete
                                diff_hit-=1
                    else:                 #MISS
                        for missClass in class_Store:
                            if y[NN[i]] == missClass:
                                class_Store[missClass][0] += 1
                                if x[inst][feature]!=x[NN[i]][feature]:
                                    if data.attributeInfo[feature][0]: #Continuous Attribute
                                        class_Store[missClass][1]+=abs(x[inst][feature]-x[NN[i]][feature])/(max_bound-min_bound)
                                    else:#Discrete
                                        class_Store[missClass][1]+=1
                                        
            #Corrects for both multiple classes, as well as missing data.
            missSum = 0 
            for each in class_Store:
                missSum += class_Store[each][0]
            missAverage = missSum/float(len(class_Store))
            
            hit_proportion=count_hit/float(len(NN)) #Correcting for Missing Data.
            for each in class_Store:
                diff_miss += (multiclass_map[each]/float(missClassPSum))*class_Store[each][1]
                
            diff = diff_miss*hit_proportion
            miss_proportion=missAverage/float(len(NN))
            diff += diff_hit*miss_proportion
                    
        else: #Binary Class Problem
            for i in range(len(NN)):  #for all nearest neighbors
                if x[inst][feature]!=data.labelMissingData and x[NN[i]][feature]!=data.labelMissingData: # add appropriate normalization.
                    
                    if y[inst]==y[NN[i]]: #HIT
                        count_hit+=1
                        if x[inst][feature]!=x[NN[i]][feature]:
                            if data.attributeInfo[feature][0]: #Continuous Attribute
                                diff_hit-=abs(x[inst][feature]-x[NN[i]][feature])/(max_bound-min_bound)
                            else:#Discrete
                                diff_hit-=1
                    else: #MISS
                        count_miss+=1
                        if x[inst][feature]!=x[NN[i]][feature]:
                            if data.attributeInfo[feature][0]: #Continuous Attribute
                                diff_miss+=abs(x[inst][feature]-x[NN[i]][feature])/(max_bound-min_bound)
                            else:#Discrete
                                diff_miss+=1 

            #Take hit/miss inbalance into account (coming from missing data)
            hit_proportion=count_hit/float(len(NN))
            miss_proportion=count_miss/float(len(NN))
            
            diff=diff_hit*miss_proportion + diff_miss*hit_proportion #applying weighting scheme to balance the scores   
        
    else: #continuous endpoint
        for i in range(len(NN)):  #for all nearest neighbors
            if x[inst][feature]!=data.labelMissingData and x[NN[i]][feature]!=data.labelMissingData: # add appropriate normalization.
                
                if abs(y[inst]-y[NN[i]])<same_class_bound: #HIT
                    count_hit+=1 
                    if x[inst][feature]!=x[NN[i]][feature]:
                        if data.attributeInfo[feature][0]: #Continuous Attribute
                            diff_hit-=abs(x[inst][feature]-x[NN[i]][feature])/(max_bound-min_bound)
                        else:#Discrete
                            diff_hit-=1
                else: #MISS
                    count_miss+=1
                    if x[inst][feature]!=x[NN[i]][feature]:
                        if data.attributeInfo[feature][0]: #Continuous Attribute
                            diff_miss+=abs(x[inst][feature]-x[NN[i]][feature])/(max_bound-min_bound)
                        else:#Discrete
                            diff_miss+=1

        #Take hit/miss inbalance into account (coming from missing data, or inability to find enough continuous neighbors)
        hit_proportion=count_hit/float(len(NN))
        miss_proportion=count_miss/float(len(NN))
        
        diff=diff_hit*miss_proportion + diff_miss*hit_proportion #applying weighting scheme to balance the scores   
  
    return diff


def calculate_distance(a,b,data):
    """ Calculates the distance between two instances in the dataset.  Handles discrete and continuous attributes. Continuous attributes are accomodated
    by scaling the distance difference within the context of the observed attribute range. If a respective data point is missing from either instance, it is left out 
    of the distance calculation. """
    d=0 #distance
    for i in range(data.numAttributes):
        if a[i]!=data.labelMissingData and b[i]!=data.labelMissingData: 
            if not data.attributeInfo[i][0]: #Discrete Attribute
                if a[i] != b[i]:
                    d+=1
            else: #Continuous Attribute
                min_bound=float(data.attributeInfo[i][1][0])
                max_bound=float(data.attributeInfo[i][1][1])
                d+=abs(float(a[i])-float(b[i]))/float(max_bound-min_bound) #Kira & Rendell, 1992 -handling continiuous attributes
    return d

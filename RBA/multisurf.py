"""
Name:        MultiSURF.py
Authors:     Gediminas Bertasius and Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     December 4, 2013
Modified:    August 25,2014
Description: 
             
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

def Run_MultiSURF(data):
    """  Called to run the MultiSURF algorithm.  
    #PARAM x- is a matrix containing the attributes of all instances in the dataset 
    #PARAM y- is a matrix containing the class of a data instance """
    
    x = [ row[0] for row in data.trainFormatted ] 
    y = [ row[1] for row in data.trainFormatted ] 
    
    print("Running MultiSURF Algorithm...")
    scores=MultiSURF(x,y,data)
    print("MultiSURF run complete.")
    
    return scores


def MultiSURF(x,y,data):
    """ Controls major MultiSURF loops. """
    ScoreList=[]
    for i in range(data.numAttributes): #initializing attributes' scores to 0
        ScoreList.append(0)
    
    #Precompute distances between all unique instance pairs within the dataset.
    print("Precomputing Distance Array")
    distanceArray = calculateDistanceArray(x,data)
    print("Computed")
    
    same_class_bound = data.phenSD
    
    D=[]
    avg_distances=[]
    for i in range(data.numTrainInstances):
        dist_vector=[]
        dist_vector=get_individual_distances(i,data,distanceArray)
        avg_distances.append(get_average(dist_vector))
        std_dev=get_std_dev(dist_vector,avg_distances[i])
        D.append(std_dev/2.0)
  
    for k in range(data.numAttributes):    #looping through attributes
        
        if data.attributeInfo[k][0]: #Continuous Attribute
            minA=data.attributeInfo[k][1][0]
            maxA=data.attributeInfo[k][1][1]
            
        count_hit_near=0
        count_miss_near=0
        count_hit_far=0
        count_miss_far=0
        
        diff_hit_near=0 #initializing the score to 0
        diff_miss_near=0
        diff_hit_far=0
        diff_miss_far=0
        
        for i in range(data.numTrainInstances):                     
            for j in range(i,data.numTrainInstances):
                if i!=j and x[i][k]!=data.labelMissingData and x[j][k]!=data.labelMissingData:
                    locator = [i,j]
                    locator = sorted(locator, reverse=True) #Access corect half of table (result of removed table redundancy)
                    d = distanceArray[locator[0]][locator[1]]
                    
                    if (d<avg_distances[i]-D[i]): #Near

                            if data.discretePhenotype: #discrete endpoint
                                if y[i]==y[j]: #Same Endpoint
                                    count_hit_near+=1
                                    if x[i][k]!=x[j][k]:
                                        if data.attributeInfo[k][0]: #Continuous Attribute (closer att scores for near same phen should yield larger att penalty)
                                            diff_hit_near-=(abs(x[i][k]-x[j][k])/(maxA-minA))
                                        else:#Discrete
                                            diff_hit_near-=1
                                else: #Different Endpoint
                                    count_miss_near+=1
                                    if x[i][k]!=x[j][k]:
                                        if data.attributeInfo[k][0]: #Continuous Attribute (farther att scores for near diff phen should yield larger att bonus)
                                            diff_miss_near+=abs(x[i][k]-x[j][k])/(maxA-minA)
                                        else:#Discrete
                                            diff_miss_near+=1
                            else:#continuous endpoint
                                if abs(y[i]-y[j])<same_class_bound:
                                    count_hit_near+=1 
                                    if x[i][k]!=x[j][k]:
                                        if data.attributeInfo[k][0]: #Continuous Attribute
                                            diff_hit_near-=(abs(x[i][k]-x[j][k])/(maxA-minA))
                                        else:#Discrete
                                            diff_hit_near-=1
                                else:
                                    count_miss_near+=1
                                    if x[i][k]!=x[j][k]:
                                        if data.attributeInfo[k][0]: #Continuous Attribute
                                            diff_miss_near+=abs(x[i][k]-x[j][k])/(maxA-minA)
                                        else:#Discrete
                                            diff_miss_near+=1
           
                    if (d>avg_distances[i]+D[i]): #Far
                            
                            if data.discretePhenotype: #discrete endpoint
                                if y[i]==y[j]:
                                        count_hit_far+=1
                                        if data.attributeInfo[k][0]: #Continuous Attribute
                                            diff_hit_far-=(abs(x[i][k]-x[j][k]))/(maxA-minA) #Attribute being similar is more important.
                                        else:#Discrete
                                            if x[i][k]==x[j][k]:
                                                diff_hit_far-=1
                                else:
                                        count_miss_far+=1
                                        if data.attributeInfo[k][0]: #Continuous Attribute
                                            diff_miss_far+=abs(x[i][k]-x[j][k])/(maxA-minA) #Attribute being similar is more important.

                                        else:#Discrete
                                            if x[i][k]==x[j][k]:
                                                diff_miss_far+=1
                            else:#continuous endpoint
                                if abs(y[i]-y[j])<same_class_bound:
                                        count_hit_far+=1 
                                        if data.attributeInfo[k][0]: #Continuous Attribute
                                            diff_hit_far-=(abs(x[i][k]-x[j][k]))/(maxA-minA) #Attribute being similar is more important.
                                        else:#Discrete
                                            if x[i][k]==x[j][k]:
                                                diff_hit_far-=1
                                else:
                                        count_miss_far+=1
                                        if data.attributeInfo[k][0]: #Continuous Attribute
                                            diff_miss_far+=abs(x[i][k]-x[j][k])/(maxA-minA) #Attribute being similar is more important.
                                        else:#Discrete
                                            if x[i][k]==x[j][k]:
                                                diff_miss_far+=1
                         
        hit_proportion=count_hit_near/float(count_hit_near+count_miss_near)
        miss_proportion=count_miss_near/float(count_hit_near+count_miss_near)
            
        diff=diff_hit_near*miss_proportion+diff_miss_near*hit_proportion #applying weighting scheme to balance the scores 
        
        hit_proportion=count_hit_far/float(count_hit_far+count_miss_far)
        miss_proportion=count_miss_far/float(count_hit_far+count_miss_far)
            
        diff+=diff_hit_far*miss_proportion+diff_miss_far*hit_proportion #applying weighting scheme to balance the scores 
        
        ScoreList[k]+=diff
                
    return ScoreList


def multiClassMultiSURF(x,y,data):
    """ Controls major MultiSURF loops. """
    ScoreList=[]
    for i in range(data.numAttributes): #initializing attributes' scores to 0
        ScoreList.append(0)
    
    #Precompute distances between all unique instance pairs within the dataset.
    print("Precomputing Distance Array")
    distanceArray = calculateDistanceArray(x,data)
    print("Computed")
    
    #For MulitClass Array Only
    multiclass_map = None
    if data.discretePhenotype and len(data.phenotypeList) > 2:
        multiclass_map = makeMultiClassMap(y,data)

    D=[]
    avg_distances=[]
    for i in range(data.numTrainInstances):
        dist_vector=[]
        dist_vector=get_individual_distances(i,data,distanceArray)
        avg_distances.append(get_average(dist_vector))
        std_dev=get_std_dev(dist_vector,avg_distances[i])
        D.append(std_dev/2.0)
  
    for k in range(data.numAttributes):    #looping through attributes
        
        if data.attributeInfo[k][0]: #Continuous Attribute
            minA=data.attributeInfo[k][1][0]
            maxA=data.attributeInfo[k][1][1]
            
        count_hit_near=0
        count_miss_near=0
        count_hit_far=0
        count_miss_far=0
        
        diff_hit_near=0 #initializing the score to 0
        diff_miss_near=0
        diff_hit_far=0
        diff_miss_far=0
        
        class_Store_near = makeClassPairMap(multiclass_map)
        class_Store_far = makeClassPairMap(multiclass_map)
        
        for i in range(data.numTrainInstances):                     
            for j in range(i,data.numTrainInstances):
                if i!=j and x[i][k]!=data.labelMissingData and x[j][k]!=data.labelMissingData:
                    locator = [i,j]
                    locator = sorted(locator, reverse=True) #Access corect half of table (result of removed table redundancy)
                    d = distanceArray[locator[0]][locator[1]]
                    
                    if (d<avg_distances[i]-D[i]): #Near

                        if y[i]==y[j]:
                            count_hit_near+=1
                            if x[i][k]!=x[j][k]:
                                if data.attributeInfo[k][0]: #Continuous Attribute
                                    diff_hit_near-=abs(x[i][k]-x[j][k])/(maxA-minA)
                                else:#Discrete
                                    diff_hit_near-=1
                        else:
                            count_miss_near+=1
                            locator = [y[i],y[j]]
                            locator = sorted(locator, reverse = True)
                            tempString = str(locator[0])+str(locator[1])
                            class_Store_near[tempString][0] += 1
                            if x[i][k]!=x[j][k]:
                                if data.attributeInfo[k][0]: #Continuous Attribute
                                    class_Store_near[tempString][1]+=abs(x[i][k]-x[j][k])/(maxA-minA)
                                else:#Discrete
                                    class_Store_near[tempString][1]+=1
           
                    if (d>avg_distances[i]+D[i]): #Far
                            
                        if y[i]==y[j]:
                            count_hit_far+=1
                            if data.attributeInfo[k][0]: #Continuous Attribute
                                diff_hit_far-=(1-abs(x[i][k]-x[j][k]))/(maxA-minA) #Attribute being similar is more important.
                            else:#Discrete
                                if x[i][k]==x[j][k]:
                                    diff_hit_far-=1
                        else:
                            count_miss_far+=1
                            locator = [y[i],y[j]]
                            locator = sorted(locator, reverse = True)
                            tempString = str(locator[0])+str(locator[1])
                            class_Store_far[tempString][0] += 1
                            
                            if data.attributeInfo[k][0]: #Continuous Attribute
                                class_Store_far[tempString][1]+=abs(x[i][k]-x[j][k])/(maxA-minA) #Attribute being similar is more important.

                            else:#Discrete
                                if x[i][k]==x[j][k]:
                                    class_Store_far[tempString][1]+=1    
        #Near
        missSum = 0 
        for each in class_Store_near:
            missSum += class_Store_near[each][0]
                         
        hit_proportion=count_hit_near/float(count_hit_near+count_miss_near) #Correcting for Missing Data.
        miss_proportion=count_miss_near/float(count_hit_near+count_miss_near) 
        
        for each in class_Store_near:
            diff_miss_near += (class_Store_near[each][0]/float(missSum))*class_Store_near[each][1]
        diff_miss_near = diff_miss_near * float(len(class_Store_near))

        diff = diff_miss_near*hit_proportion + diff_hit_near*miss_proportion 
                         
        #Far
        missSum = 0 
        for each in class_Store_far:
            missSum += class_Store_far[each][0]

        hit_proportion=count_hit_far/float(count_hit_far+count_miss_far) #Correcting for Missing Data.
        miss_proportion=count_miss_far/float(count_hit_far+count_miss_far) 
        
        for each in class_Store_far:
            diff_miss_far += (class_Store_far[each][0]/float(missSum))*class_Store_far[each][1]

        diff_miss_far = diff_miss_far * float(len(class_Store_far))
        
        diff += diff_miss_far*hit_proportion + diff_hit_far*miss_proportion   
           
        ScoreList[k]+=diff    
                
    return ScoreList

                    
def get_std_dev(dist_vector,avg):
    sum=0;
    for i in range(len(dist_vector)):
        sum+=(dist_vector[i]-avg)**2
    sum=sum/float(len(dist_vector))
    return (sum**0.5)


def get_average(dist_vector):
    sum=0
    for i in range(len(dist_vector)):
        sum+=dist_vector[i];
    return sum/float(len(dist_vector))


def get_individual_distances(i,data,distanceArray):
    d=[]
    for j in range(data.numTrainInstances):
        if (i!=j):
            locator = [i,j]
            locator = sorted(locator, reverse=True) #Access corect half of table (result of removed table redundancy)
            d.append(distanceArray[locator[0]][locator[1]])
    return d

        
def calculateDistanceArray(x,data):
    #make empty distance array container (we will only fill up the non redundant half of the array
    distArray = []
    for i in range(data.numTrainInstances):
        distArray.append([])
        for j in range(data.numTrainInstances):
            distArray[i].append(None)
            
    for i in range(1, data.numTrainInstances):
        for j in range(0,i):
            distArray[i][j] = calculate_distance(x[i],x[j],data)
            
    return distArray
        

def makeMultiClassMap(y, data):
    #finding number of classes in the dataset and storing them into the map
    multiclass_map={}
    
    for i in range(data.numTrainInstances):
        if (y[i] not in multiclass_map):
            multiclass_map[y[i]]=0
        else:
            multiclass_map[y[i]]+=1
            
    for each in data.phenotypeList: #For each class store probability of class occurrence in dataset.
        multiclass_map[each] = multiclass_map[each]/float(data.numTrainInstances)
      
    return multiclass_map


def makeClassPairMap(multiclass_map):
    #finding number of classes in the dataset and storing them into the map
    classPair_map={}
    for each in multiclass_map:
        for other in multiclass_map:
            if each != other:
                locator = [each,other]
                locator = sorted(locator, reverse = True)
                tempString = str(locator[0])+str(locator[1])
                if (tempString not in classPair_map):
                    classPair_map[tempString] = [0,0]
    return classPair_map


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

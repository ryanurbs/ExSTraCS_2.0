"""
Name:        ExSTraCS_ConfigParser.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: Manages the configuration file by loading, parsing, and passing it's values to ExSTraCS_Constants. Also includes a method for generating 
             datasets for cross validation.
             
---------------------------------------------------------------------------------------------------------------------------------------------------------
ExSTraCS V2.0: Extended Supervised Tracking and Classifying System - An advanced LCS designed specifically for complex, noisy classification/data mining tasks, 
such as biomedical/bioinformatics/epidemiological problem domains.  This algorithm should be well suited to any supervised learning problem involving 
classification, prediction, data mining, and knowledge discovery.  This algorithm would NOT be suited to function approximation, behavioral modeling, 
or other multi-step problems.  This LCS algorithm is most closely based on the "UCS" algorithm, an LCS introduced by Ester Bernado-Mansilla and 
Josep Garrell-Guiu (2003) which in turn is based heavily on "XCS", an LCS introduced by Stewart Wilson (1995).  

Copyright (C) 2014 Ryan Urbanowicz 
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, 
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
#Import Required Modules-------------------------------
from exstracs_constants import *
import os
import copy
import random
#------------------------------------------------------

class ConfigParser:
    def __init__(self, filename):
        random.seed(1) #Same Random Seed always used here, such that the same CV datasets will be generated if run again.
        self.commentChar = '#'
        self.paramChar =  '='
        self.parameters = self.parseConfig(filename) #Parse the configuration file and get all parameters.
        cons.setConstants(self.parameters) #Begin building constants Class using parameters from configuration file
        
        if cons.internalCrossValidation == 0 or cons.internalCrossValidation == 1: 
            pass
        else: #Do internal CV
            self.CVPart()
        
 
    def parseConfig(self, filename):
        """ Parse the Configuration File"""
        parameters = {}
        try:
            f = open(filename,'rU')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', filename)
            raise 
        else:
            for line in f:
                # First, remove comments:
                if self.commentChar in line:
                    # split on comment char, keep only the part before
                    line, comment = line.split(self.commentChar, 1)
                # Second, find lines with an parameter=value:
                if self.paramChar in line:
                    # split on parameter char:
                    parameter, value = line.split(self.paramChar, 1)
                    # strip spaces:
                    parameter = parameter.strip()
                    value = value.strip()
                    # store in dictionary:
                    parameters[parameter] = value
                    
            f.close()
        return parameters
    
    
    def CVPart(self):
        """ Given a data set, CVPart randomly partitions it into X random balanced 
        partitions for cross validation which are individually saved in the specified file. 
        filePath - specifies the path and name of the new datasets. """
        numPartitions = cons.internalCrossValidation
        folderName = copy.deepcopy(cons.trainFile)
        fileName = folderName.split('\\')
        fileName = fileName[len(fileName)-1]
        filePath = folderName+'\\'+fileName

        #Make folder for CV Datasets
        if not os.path.exists(folderName):
            os.mkdir(folderName)     
            
        # Open the specified data file.
        try:
            f = open(cons.trainFile+'.txt', 'rU')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', cons.trainFile+'.txt')
            raise 
        else:
            datasetList = []
            headerList = f.readline().rstrip('\n').split('\t')  #strip off first row
            for line in f:
                lineList = line.strip('\n').split('\t')
                datasetList.append(lineList)
            f.close()
        dataLength = len(datasetList)   
            
        #Characterize Phenotype----------------------------------------------------------------------------
        discretePhenotype = True
        if cons.labelPhenotype in headerList:
            phenotypeRef = headerList.index(cons.labelPhenotype)
        else:
            print("Error: ConfigParser - Phenotype Label not found.")

        inst = 0
        classDict = {}
        while len(list(classDict.keys())) <= cons.discreteAttributeLimit and inst < dataLength:  #Checks which discriminate between discrete and continuous attribute
            target = datasetList[inst][phenotypeRef]
            if target in list(classDict.keys()):  #Check if we've seen this attribute state yet.
                classDict[target] += 1
            else: #New state observed
                classDict[target] = 1
            inst += 1
            
        if len(list(classDict.keys())) > cons.discreteAttributeLimit:
            discretePhenotype = False
        else:
            pass
        #---------------------------------------------------------------------------------------------------
        
        CVList = [] #stores all partitions
        for x in range(numPartitions):
            CVList.append([])
        
        if discretePhenotype:
            masterList = []
            classKeys = list(classDict.keys())
            for i in range(len(classKeys)):
                masterList.append([])
            for i in datasetList:
                notfound = True
                j = 0
                while notfound:
                    if i[phenotypeRef] == classKeys[j]:
                        masterList[j].append(i)
                        notfound = False
                    j += 1
            
            #Randomize class instances before partitioning------------------
            from random import shuffle
            for i in range(len(classKeys)):
                shuffle(masterList[i])
            #---------------------------------------------------------------
                
            for currentClass in masterList:
                currPart = 0
                counter = 0
                for x in currentClass:
                    CVList[currPart].append(x)
                    counter += 1
                    currPart = counter%numPartitions
                    
            self.makePartitions(CVList,numPartitions,filePath,headerList)
            
        else: #Continuous Endpoint
            from random import shuffle
            shuffle(datasetList)  
            currPart = 0
            counter = 0
            for x in datasetList:
                CVList[currPart].append(x)
                counter += 1
                currPart = counter%numPartitions
            
            self.makePartitions(CVList,numPartitions,filePath,headerList)
            
            
    def makePartitions(self,CVList,numPartitions,filePath,headerList):         
        for part in range(numPartitions): #Builds CV data files.
            if not os.path.exists(filePath+'_CV_'+str(part)+'_Train.txt') or not os.path.exists(filePath+'_CV_'+str(part)+'_Test.txt'):
                print("Making new CV files:  "+filePath+'_CV_'+str(part))
                trainFile = open(filePath+'_CV_'+str(part)+'_Train.txt','w')
                testFile = open(filePath+'_CV_'+str(part)+'_Test.txt','w')
                
                for i in range(len(headerList)):   
                    if i < len(headerList)-1:
                        testFile.write(headerList[i] + "\t")
                        trainFile.write(headerList[i] + "\t")  
                    else:
                        testFile.write(headerList[i] + "\n")
                        trainFile.write(headerList[i] + "\n") 
    
                testList=CVList[part]
                trainList=[]
                tempList = []                 
                for x in range(numPartitions): 
                    tempList.append(x)                            
                tempList.pop(part)
    
                for v in tempList: #for each training partition
                    trainList.extend(CVList[v])    
            
                for i in testList: #Write to Test Datafile
                    tempString = ''
                    for point in range(len(i)):
                        if point < len(i)-1:
                            tempString = tempString + str(i[point])+"\t"
                        else:
                            tempString = tempString +str(i[point])+"\n"                        
                    testFile.write(tempString)
                          
                for i in trainList: #Write to Train Datafile
                    tempString = ''
                    for point in range(len(i)):
                        if point < len(i)-1:
                            tempString = tempString + str(i[point])+"\t"
                        else:
                            tempString = tempString +str(i[point])+"\n"                        
                    trainFile.write(tempString)
                                                    
                trainFile.close()
                testFile.close()  

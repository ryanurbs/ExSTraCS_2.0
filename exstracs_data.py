"""
Name:        ExSTraCS_DataManagement.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: Loads the dataset, characterizes and stores critical features of the datasets (including discrete vs. continuous attributes and phenotype), handles missing 
             data, and finally formats the data so that it may be conveniently utilized by ExSTraCS.
             
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
import math
import random
import sys
from exstracs_constants import * 
#------------------------------------------------------

class DataManagement:
    def __init__(self, trainFile, testFile, infoList = None):
        #Set random seed if specified.-----------------------------------------------
        if cons.useSeed:
            random.seed(cons.randomSeed)
        else:
            random.seed(None)
            
        if cons.offlineData:
            #Initialize global variables-------------------------------------------------
            self.numAttributes = None  # Saves the number of attributes in the input file. 
            self.areInstanceIDs = False
            self.instanceIDRef = None
            self.phenotypeRef = None
            self.discretePhenotype = True 
            self.attributeInfo = [] #Stores Discrete (0) vs. Continuous (1)
            self.phenotypeList = [] #stores discrete phenotype values OR for continuous phenotype, max and min values
            self.phenotypeRandomPick = None #Used to approximate random phenotype selection accuracy.
            self.phenotypeRange = None
            self.phenSD = None
            self.labelMissingData = cons.labelMissingData
            self.missingEndpointList = []
            
            #Train/Test Specific-----------------------------------------------------------------------------
            self.trainHeaderList = []
            self.testHeaderList = []
            self.numTrainInstances = None 
            self.numTestInstances = None 
            self.averageStateCount = None     
            
            self.discreteCount = 0
            self.continuousCount = 0
            self.classCount = {}
            self.classPredictionWeights = {}
            #Detect Features of training data--------------------------------------------------------------------------
            print("----------------------------------------------------------------------------")
            print("Environment: Formatting Data... ")
            
            rawTrainData = self.loadData(trainFile+'.txt', True) #Load the raw data.
    
            self.characterizeDataset(rawTrainData)  #Detect number of attributes, instances, and reference locations.
            
            if cons.testFile == 'None': #If no testing data is available, formatting relies solely on training data.
                data4Formating = rawTrainData
            else:
                rawTestData = self.loadData(testFile+'.txt', False) #Load the raw data.
                self.compareDataset(rawTestData) #Ensure that key features are the same between training and testing datasets.
    
            self.discriminatePhenotype(rawTrainData) #Determine if endpoint/phenotype is discrete or continuous.
            if self.discretePhenotype:
                self.discriminateClasses(rawTrainData) #Detect number of unique phenotype identifiers.
            else:
                print("DataManagement - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
                
            self.discriminateAttributes(rawTrainData) #Detect whether attributes are discrete or continuous.
            self.characterizeAttributes(rawTrainData) #Determine potential attribute states or ranges.
            
            #Rule Specificity Limit (RSL) ----------------------------------------------------------------------------
            if cons.RSL_Override > 0:
                self.specLimit = cons.RSL_Override
            else:
                #Calculate Rule Specificity Limit --------------------------------------------------------------------
                print("DataManagement: Estimating Classifier Specification Limit")
                i = 1
                uniqueCombinations = math.pow(self.averageStateCount, i)
                while uniqueCombinations < self.numTrainInstances:
                    i += 1
                    uniqueCombinations = math.pow(self.averageStateCount,i)
                self.specLimit = i
                if self.numAttributes < self.specLimit:  #Never allow the specLimit to be larger than the number of attributes in the dataset.
                    self.specLimit = self.numAttributes
            print("DataManagement: Specification Limit = "+str(self.specLimit))
            
            #Format and Shuffle Datasets----------------------------------------------------------------------------------------
            if cons.testFile != 'None':
                self.testFormatted = self.formatData(rawTestData, False) #Stores the formatted testing data set used throughout the algorithm.

            self.trainFormatted = self.formatData(rawTrainData, True) #Stores the formatted training data set used throughout the algorithm.       
            print("----------------------------------------------------------------------------")
        else:
            #Initialize global variables-------------------------------------------------
            self.numAttributes = infoList[0]  # Saves the number of attributes in the input file. 
            self.areInstanceIDs = False
            self.instanceIDRef = None
            self.phenotypeRef = None
            self.discretePhenotype = infoList[1]
            self.attributeInfo = infoList[2] #Stores Discrete (0) vs. Continuous (1)
            self.phenotypeList = infoList[3] #stores discrete phenotype values OR for continuous phenotype, max and min values
            self.phenotypeRange = infoList[4]
            self.trainHeaderList = infoList[5]
            self.numTrainInstances = infoList[6]
            self.specLimit = 7
        
    def loadData(self, dataFile, doTrain):
        """ Load the data file. """     
        print("DataManagement: Loading Data... " + str(dataFile))
        datasetList = []
        try:       
            f = open(dataFile,'rU')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', dataFile)
            raise 
        else:
            if doTrain:
                self.trainHeaderList = f.readline().rstrip('\n').split('\t')   #strip off first row
            else:
                self.testHeaderList = f.readline().rstrip('\n').split('\t')   #strip off first row
            for line in f:
                lineList = line.strip('\n').split('\t')
                datasetList.append(lineList)
            f.close()

        return datasetList
    
    
    def characterizeDataset(self, rawTrainData):
        " Detect basic dataset parameters " 
        #Detect Instance ID's and save location if they occur.
        if cons.labelInstanceID in self.trainHeaderList:
            self.areInstanceIDs = True
            self.instanceIDRef = self.trainHeaderList.index(cons.labelInstanceID)
            print("DataManagement: Instance ID Column location = "+str(self.instanceIDRef))

            self.numAttributes = len(self.trainHeaderList)-2 #one column for InstanceID and another for the phenotype.
        else:
            self.numAttributes = len(self.trainHeaderList)-1
            
        if cons.labelPhenotype in self.trainHeaderList:
            self.phenotypeRef = self.trainHeaderList.index(cons.labelPhenotype)

            print("DataManagement: Phenotype Column Location = "+str(self.phenotypeRef))
        else:
            print("DataManagement: Error - Phenotype column not found!  Check data set to ensure correct phenotype column label, or inclusion in the data.")

        if self.areInstanceIDs:
            if self.phenotypeRef > self.instanceIDRef:
                self.trainHeaderList.pop(self.phenotypeRef)
                self.trainHeaderList.pop(self.instanceIDRef)
            else:
                self.trainHeaderList.pop(self.instanceIDRef)
                self.trainHeaderList.pop(self.phenotypeRef)
        else:
            self.trainHeaderList.pop(self.phenotypeRef)
            
        self.numTrainInstances = len(rawTrainData)
        print("DataManagement: Number of Attributes = " + str(self.numAttributes)) #DEBUG
        print("DataManagement: Number of Instances = " + str(self.numTrainInstances)) #DEBUG


    def discriminatePhenotype(self, rawData):
        """ Determine whether phenotype is Discrete(classes) or Continuous """
        print("DataManagement: Analyzing Phenotype...")
        inst = 0
        classDict = {}
        while len(list(classDict.keys())) <= cons.discreteAttributeLimit and inst < self.numTrainInstances:  #Checks which discriminate between discrete and continuous attribute
            target = rawData[inst][self.phenotypeRef]
            if target in list(classDict.keys()):  #Check if we've seen this attribute state yet.
                classDict[target] += 1
            elif target == cons.labelMissingData: #Ignore data rows with missing endpoint information.
                self.missingEndpointList.append(inst)
            else: #New state observed
                classDict[target] = 1
            inst += 1

        if len(list(classDict.keys())) > cons.discreteAttributeLimit:
            self.discretePhenotype = False
            self.phenotypeList = [float(target),float(target)]
            print("DataManagement: Phenotype Detected as Continuous.")
        else:
            print("DataManagement: Phenotype Detected as Discrete.")
            
    
    def discriminateClasses(self, rawData):
        """ Determines number of classes and their identifiers. Only used if phenotype is discrete. Requires both training and testing datasets in order to standardize formatting across both. """
        print("DataManagement: Detecting Classes...")
        inst = 0
        while inst < self.numTrainInstances:
            target = rawData[inst][self.phenotypeRef]
            if target in self.phenotypeList:
                self.classCount[target] += 1 #NOTE: Could potentially store state frequency information to guide learning.
                self.classPredictionWeights[target] += 1
            elif target == cons.labelMissingData: #Ignore missing data
                pass
            else:
                self.phenotypeList.append(target)
                self.classCount[target] = 1
                self.classPredictionWeights[target] = 1
            inst += 1
        print("DataManagement: Following Classes Detected:")
        print(self.phenotypeList)
        total = 0
        for each in list(self.classCount.keys()):
            total += self.classCount[each]
            print("Class: "+str(each)+ " count = "+ str(self.classCount[each]))
            
        for each in list(self.classCount.keys()):
            self.classPredictionWeights[each] = 1- (self.classPredictionWeights[each] /float(total))
        print(self.classPredictionWeights)
        
        #Random Selection Determination (Not specifically adapted for class imbalance)
        self.phenotypeRandomPick = 1 / float(len(self.phenotypeList))
            
                     
    def compareDataset(self, rawTestData):
        " Ensures that key dataset parameters are indeed the same for training and testing datasets "
        if self.areInstanceIDs:
            if self.phenotypeRef > self.instanceIDRef:
                self.testHeaderList.pop(self.phenotypeRef)
                self.testHeaderList.pop(self.instanceIDRef)
            else:
                self.testHeaderList.pop(self.instanceIDRef)
                self.testHeaderList.pop(self.phenotypeRef)
        else:
            self.testHeaderList.pop(self.phenotypeRef)
            
        if self.trainHeaderList != self.testHeaderList:
            print("DataManagement: Error - Training and Testing Dataset Headers are not equivalent")

        self.numTestInstances = len(rawTestData)
        print("DataManagement: Number of Attributes = " + str(self.numAttributes)) #DEBUG
        print("DataManagement: Number of Instances = " + str(self.numTestInstances)) #DEBUG



    def discriminateAttributes(self, rawData):
        """ Determine whether attributes are Discrete or Continuous. Requires both training and testing datasets in order to standardize formatting across both. """
        print("DataManagement: Detecting Attributes...")
        self.discreteCount = 0
        self.continuousCount = 0
        for att in range(len(rawData[0])):
            if att != self.instanceIDRef and att != self.phenotypeRef:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                attIsDiscrete = True
                inst = 0
                stateDict = {}
                while len(list(stateDict.keys())) <= cons.discreteAttributeLimit and inst < self.numTrainInstances:  #Checks which discriminate between discrete and continuous attribute
                    if inst in self.missingEndpointList: #don't use training instances without endpoint information.
                        inst += 1
                        pass
                    else:
                        target = rawData[inst][att]
                        if target in list(stateDict.keys()):  #Check if we've seen this attribute state yet.
                            stateDict[target] += 1
                        elif target == cons.labelMissingData: #Ignore missing data
                            pass
                        else: #New state observed
                            stateDict[target] = 1
                        inst += 1

                if len(list(stateDict.keys())) > cons.discreteAttributeLimit:
                    attIsDiscrete = False
                if attIsDiscrete:
                    self.attributeInfo.append([0,[]])    
                    self.discreteCount += 1
                else:
                    self.attributeInfo.append([1,[float(target),float(target)]])   #[min,max]
                    self.continuousCount += 1
        print("DataManagement: Identified "+str(self.discreteCount)+" discrete and "+str(self.continuousCount)+" continuous attributes.") #Debug

            
    def characterizeAttributes(self, rawData):
        """ Determine range or states of each attribute.  Requires both training and testing datasets in order to standardize formatting across both. """
        print("DataManagement: Characterizing Attributes...")
        attributeID = 0
        self.averageStateCount = 0
        for att in range(len(rawData[0])):
            if att != self.instanceIDRef and att != self.phenotypeRef:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                for inst in range(len(rawData)):
                    if inst in self.missingEndpointList: #don't use training instances without endpoint information.
                        pass
                    else:
                        target = rawData[inst][att]
                        if not self.attributeInfo[attributeID][0]: #If attribute is discrete
                            if target in self.attributeInfo[attributeID][1] or target == cons.labelMissingData:
                                pass  #NOTE: Could potentially store state frequency information to guide learning.
                            else:
                                self.attributeInfo[attributeID][1].append(target)
                                self.averageStateCount += 1
                        else: #If attribute is continuous
                            #Find Minimum and Maximum values for the continuous attribute so we know the range.
                            if target == cons.labelMissingData:
                                pass
                            elif float(target) > self.attributeInfo[attributeID][1][1]:  #error
                                self.attributeInfo[attributeID][1][1] = float(target)
                            elif float(target) < self.attributeInfo[attributeID][1][0]:
                                self.attributeInfo[attributeID][1][0] = float(target)
                            else:
                                pass
                if self.attributeInfo[attributeID][0]: #If attribute is continuous
                    self.averageStateCount += 2 #Simplify continuous attributes to be counted as two-state variables (high/low) for specLimit calculation.
                attributeID += 1
        self.averageStateCount = self.averageStateCount / float(self.numAttributes)


    def calcSD(self, phenList):
        """  Calculate the standard deviation of the continuous phenotype scores. """
        for i in range(len(phenList)):
            phenList[i] = float(phenList[i])

        avg = float(sum(phenList)/len(phenList))
        dev = []
        for x in phenList:
            dev.append(x-avg)
            sqr = []
        for x in dev:
            sqr.append(x*x)
            
        return math.sqrt(sum(sqr)/(len(sqr)-1))
     
            
    def formatData(self,rawData,training):
        """ Get the data into a format convenient for the algorithm to interact with. Our format is consistent with our rule representation, namely, Attribute-list knowledge representation (ALKR),"""
        formatted = []
        testMissingEndpoints = []
        #Initialize data format---------------------------------------------------------
        for i in range(len(rawData)):  
            formatted.append([None,None,None]) #[Attribute States, Phenotype, InstanceID]

        for inst in range(len(rawData)):
            stateList = []
            attributeID = 0
            for att in range(len(rawData[0])):
                if att != self.instanceIDRef and att != self.phenotypeRef:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                    target = rawData[inst][att]
                    
                    if self.attributeInfo[attributeID][0]: #If the attribute is continuous
                        if target == cons.labelMissingData:
                            stateList.append(target) #Missing data saved as text label
                        else:
                            stateList.append(float(target)) #Save continuous data as floats. 
                    else: #If the attribute is discrete - Format the data to correspond to the GABIL (DeJong 1991)
                        stateList.append(target) #missing data, and discrete variables, all stored as string objects   
                    attributeID += 1
            
            #Final Format-----------------------------------------------
            formatted[inst][0] = stateList                           #Attribute states stored here
            if self.discretePhenotype:
                if not training: #Testing Data Check for Missing Endpoints to exclude from analysis
                    if rawData[inst][self.phenotypeRef] == self.labelMissingData:
                        testMissingEndpoints.append(inst)
                formatted[inst][1] = rawData[inst][self.phenotypeRef]        #phenotype stored here
            else:
                print("DataManagement - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
            if self.areInstanceIDs:
                formatted[inst][2] = rawData[inst][self.instanceIDRef]   #Instance ID stored here
            else: #An instance ID is required to tie instances to attribute tracking scores
                formatted[inst][2] = inst #NOTE ID's are assigned before shuffle - id's capture order of instances in original dataset file.
            #-----------------------------------------------------------
        if training:
            #Remove instances without endpoint information.  We do this here so that automatically added instance identifiers still correspond to original dataset.
            if len(self.missingEndpointList) > 0:
                self.missingEndpointList.reverse() #Remove from last to first to avoid problems.
                for each in self.missingEndpointList:
                    formatted.pop(each)
                self.numTrainInstances = self.numTrainInstances - len(self.missingEndpointList) #Correct number of training instances based on number of instances with missing endpoints.
                print("DataManagement: Adjusted Number of Training Instances = " + str(self.numTrainInstances)) #DEBUG
            random.shuffle(formatted) #One time randomization of the order the of the instances in the data, so that if the data was ordered by phenotype, this potential learning bias (based on instance ordering) is eliminated.  
        else:
            if len(testMissingEndpoints) > 0:
                testMissingEndpoints.reverse() #Remove from last to first to avoid problems.
                for each in testMissingEndpoints:
                    formatted.pop(each)
                self.numTestInstances = self.numTestInstances - len(testMissingEndpoints) #Correct number of training instances based on number of instances with missing endpoints.
                print("DataManagement: Adjusted Number of Testing Instances = " + str(self.numTestInstances)) #DEBUG
        return formatted
    
    
    def saveTempTurfData(self):
        """  Store and preserve original dataset formatting for TuRF EK generation. """
        self.turfformatted = copy.deepcopy(self.trainFormatted)
        self.turfHeaderList = copy.deepcopy(self.trainHeaderList)
        self.turfNumAttributes = copy.deepcopy(self.numAttributes)
        self.tierList = [] #will store attribute names from headerList
        
        
    def returntoFullData(self):
        """ Following TuRF completion, return to orignal complete dataset. """
        self.trainFormatted = self.turfformatted
        self.trainHeaderList = self.turfHeaderList
        self.numAttributes = self.turfNumAttributes
        
    
    def turfDataManagement(self, filterScores, turfPercent):
        """ Add 'Turf' wrapper to any Relief Based algorithm, so that the respective algorithm is run iteratively, each iteration removing 
        a percentage of attributes from consideration, for recalculation of remaining attribute scores. For example, the ReliefF algorithm 
        with this wrapper is called Turf, The SURF algorithm with this wrapper is called SURFnTurf.  The SURF* algorithm with this wrapper 
        is called SURF*nTurf."""
        #Determine number of attributes to remove.
        numRemove = int(self.numAttributes*turfPercent)
        print("Removing "+str(numRemove)+" attribute(s).")
        
        currentFilteredList = []
        #Iterate through data removing lowest each time.
        for i in range(0, numRemove):
            lowVal = min(filterScores)
            lowRef = filterScores.index(lowVal)
            currentFilteredList.append(self.trainHeaderList.pop(lowRef))
            self.numAttributes -= 1
            for k in range(self.numTrainInstances):
                self.trainFormatted[k][0].pop(lowRef)
            filterScores.pop(lowRef)

        self.tierList.append(currentFilteredList) #store filtered attributes as list of removed levels.
        random.shuffle(self.trainFormatted) #Only makes a difference if a subset of instances is being used for calculations, this way a different subset will be used each time.

        print(str(self.numAttributes) + " remaining after turf iteration.")
        
        if self.numAttributes*float(turfPercent) < 1: #Prevent iterations that do not remove attributes (useful for smaller datasets)
            keepGoing = False
        else:
            keepGoing = True
            
        return keepGoing
    

    def makeFilteredDataset(self, attsInData, fileName, filterScores):
        """ Makes a new dataset, which has filtered out the lowest scoring attributes ( """
        if attsInData > self.numAttributes:
            print("NOTICE: Requested number of attributes ("+str(attsInData)+" in dataset not available.  Returning total number of available attributes instead. ("+str(self.numAttributes)+")")
            attsInData = self.numAttributes
        
        try:  
            dataOut = open(fileName+'_filtered.txt','w') 
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', fileName+'_filtered.txt')
            raise 

        if attsInData < self.numAttributes:
            numRemove = self.numAttributes - attsInData
        else:
            numRemove = 0
        
        #Iterate through data removing lowest each time.
        for i in range(0, numRemove):
            lowRef = 0
            lowVal = filterScores[0]
            for j in range(1,self.numAttributes):
                if filterScores[j] < lowVal:
                    lowVal = filterScores[j]
                    lowRef = j
            #Lowest Value found
            self.trainHeaderList.pop(lowRef)
            self.testHeaderList.pop(lowRef)
            self.attributeInfo.pop(lowRef)
            self.numAttributes -= 1
            for k in range(self.numTrainInstances):
                self.trainFormatted[k][0].pop(lowRef)
            for k in range(self.numTestInstances):
                self.testFormatted[k][0].pop(lowRef)
                
        #numAttributes is now equal to the filtered attribute number specified.
        for i in range(self.numAttributes):
            dataOut.write(self.trainHeaderList[i]+'\t')
        dataOut.write('Class'+'\n')
        
        for i in range(self.numTrainInstances):
            for j in range(self.numAttributes):
                dataOut.write(str(self.trainFormatted[i][0][j])+'\t')
            dataOut.write(str(self.trainFormatted[i][1])+'\n')

        dataOut.close()
  
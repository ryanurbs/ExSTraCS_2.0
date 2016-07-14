"""
Name:        ExSTraCS_Constants.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: Stores and makes available all alogrithmic run parameters, and acts as a gateway for referencing the timer, environment, dataset properties, 
             attribute tracking, and expert knowledge scores/weights.  This is also where the generation expert knowledge and respective weights is controlled.
             
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
import copy
import os
import time
#------------------------------------------------------

class Constants:
    def setConstants(self,par):
        """ Takes the parameters parsed as a dictionary in ExSTraCS_ConfigParser and makes these parameters available throughout ExSTraCS. 
        Default values are provided for some parameters through the use of try/except commands so that users can generate simpler
        configuration files that only ."""
        try:
            extCheck = par['trainFile'][len(par['trainFile'])-4:len(par['trainFile'])] #Check for included .txt file extension
            if extCheck == '.txt':
                self.trainFile = par['trainFile'][0:len(par['trainFile'])-4]
            else:
                self.trainFile = par['trainFile']                                           #Saved as text
        except:
            print('Constants: Error - Default value not available for "trainFile", please specify value in the configuration file.')
        try:
            extCheck = par['testFile'][len(par['testFile'])-4:len(par['testFile'])]  #Check for included .txt file extension
            if extCheck == '.txt':
                self.testFile = par['testFile'][0:len(par['testFile'])-4]
            else:
                self.testFile = par['testFile']                                             #Saved as text
        except:
            self.testFile = 'None'
        trainName = self.trainFile.split('/')
        trainName = trainName[len(trainName)-1] #Grab FileName only.
        try:
            if str(par['outFileName']) == 'None' or str(par['outFileName']) == 'none':
                self.originalOutFileName = trainName                               #Saved as text
                self.outFileName = trainName +'_ExSTraCS'                          #Saved as text
            else:
                self.originalOutFileName = str(par['outFileName'])+trainName           #Saved as text
                self.outFileName = str(par['outFileName'])+trainName+'_ExSTraCS'                      #Saved as text
        except:
            self.originalOutFileName = trainName                               #Saved as text
            self.outFileName = trainName +'_ExSTraCS'                          #Saved as text
        try:
            self.offlineData = bool(int(par['offlineData']))                        #Saved as Boolean
        except: #Default
            self.offlineData = True
        try:
            self.internalCrossValidation = int(par['internalCrossValidation'])      #Saved as int
        except: #Default
            self.internalCrossValidation = 0
            
        try:
            if par['randomSeed'] == 'False' or par['randomSeed'] == 'false':
                self.useSeed = False
            else:
                self.useSeed = True
                self.randomSeed = int(par['randomSeed'])                            #Saved as int
        except: #Default
            self.useSeed = False
              
        #----------------------------------------------------------------------------------
        try:
            self.labelInstanceID = par['labelInstanceID']                           #Saved as text
        except: #Default
            self.labelInstanceID = 'InstanceID'
        try:
            self.labelPhenotype = par['labelPhenotype']                             #Saved as text
        except: #Default
            self.labelPhenotype = 'Class'
        try:
            self.discreteAttributeLimit = int(par['discreteAttributeLimit'])        #Saved as int
        except: #Default
            self.discreteAttributeLimit = 10
        try:
            self.labelMissingData = par['labelMissingData']                         #Saved as text
        except: #Default
            self.labelMissingData = 'NA'
        try:
            self.outputSummary = bool(int(par['outputSummary']))                    #Saved as Boolean
        except: #Default
            self.outputSummary = True
        try:
            self.outputPopulation = bool(int(par['outputPopulation']))              #Saved as Boolean
        except: #Default
            self.outputPopulation = True
        try:
            self.outputAttCoOccur = bool(int(par['outputAttCoOccur']))              #Saved as Boolean  
        except: #Default
            self.outputAttCoOccur = True     
        try:
            self.outputTestPredictions = bool(int(par['outputTestPredictions']))              #Saved as Boolean  
        except: #Default
            self.outputTestPredictions = True  
        try:
            self.onlyTest = bool(int(par['onlyTest']))              #Saved as Boolean  
        except: #Default
            self.onlyTest = False

        #----------------------------------------------------------------------------------
        try:
            self.trackingFrequency = int(par['trackingFrequency'])      #Saved as int
        except: #Default
            self.trackingFrequency = 0
        try:
            self.learningIterations = par['learningIterations']         #Saved as text
        except: #Default
            self.learningIterations ='5000.10000.20000.100000'
        try:
            self.N = int(par['N'])                                      #Saved as int
        except: #Default
            self.N = 1000
        try:
            self.nu = int(par['nu'])                                    #Saved as int
        except: #Default
            self.nu = 1
        try:
            self.chi = float(par['chi'])                                #Saved as float
        except: #Default
            self.chi = 0.8
        try:
            self.upsilon = float(par['upsilon'])                        #Saved as float
        except: #Default
            self.upsilon = 0.04
        try:
            self.theta_GA = int(par['theta_GA'])                        #Saved as int
        except: #Default
            self.theta_GA = 25
        try:
            self.theta_del = int(par['theta_del'])                      #Saved as int
        except: #Default
            self.theta_del = 20
        try:
            self.theta_sub = int(par['theta_sub'])                      #Saved as int
        except: #Default
            self.theta_sub = 20
        try:
            self.acc_sub = float(par['acc_sub'])                        #Saved as float
        except: #Default
            self.acc_sub = 0.99
        try:
            self.beta = float(par['beta'])                              #Saved as float
        except: #Default
            self.beta = 0.2
        try:
            self.delta = float(par['delta'])                            #Saved as float
        except: #Default
            self.delta = 0.1
        try:
            self.init_fit = float(par['init_fit'])                      #Saved as float
        except: #Default
            self.init_fit = 0.01
        try:
            self.fitnessReduction = float(par['fitnessReduction'])      #Saved as float
        except: #Default
            self.fitnessReduction = 0.1
        try:
            self.theta_sel = float(par['theta_sel'])                    #Saved as float
        except: #Default
            self.theta_sel = 0.5
        try:
            self.RSL_Override = int(par['RSL_Override'])                    #Saved as float
        except: #Default
            self.RSL_Override = 0
        
        try:
            self.doSubsumption = bool(int(par['doSubsumption']))                #Saved as Boolean
        except: #Default
            self.doSubsumption = True
        try:
            self.selectionMethod = par['selectionMethod']                       #Saved as text
        except: #Default
            self.selectionMethod = 'tournament'
        
        try:
            self.doAttributeTracking = bool(int(par['doAttributeTracking']))    #Saved as Boolean
        except: #Default
            self.doAttributeTracking = True
        try:
            self.doAttributeFeedback = bool(int(par['doAttributeFeedback']))    #Saved as Boolean
        except: #Default
            self.doAttributeFeedback = True
            
        #Expert Knowledge Parameters -----------------------------------------------------------------------
        try:
            self.useExpertKnowledge = bool(int(par['useExpertKnowledge']))          #Saved as Boolean
        except: #Default
            self.useExpertKnowledge = True
            
        if self.useExpertKnowledge:
            try:
                if str(par['external_EK_Generation']) == 'None' or str(par['external_EK_Generation']) == 'none':
                    self.internal_EK_Generation = True
                    try:
                        if par['outEKFileName'] == 'None' or par['outEKFileName'] == 'none':
                            self.outEKFileName = trainName 
                            self.originalOutEKFileName = trainName
                        else:
                            self.outEKFileName = par['outEKFileName']+trainName          #Saved as text
                            self.originalOutEKFileName = par['outEKFileName']+trainName
                    except:
                        self.outEKFileName = trainName 
                        self.originalOutEKFileName = trainName
                    
                else:
                    self.internal_EK_Generation = False
                    try:
                        self.EK_source = str(par['external_EK_Generation'])  #Saved as text
                    except:
                        print('Constants: Error - No default available for external EK file.')
     
            except: #Default - external_EK_Generation is not specified.
                self.internal_EK_Generation = True
                try:
                    if par['outEKFileName'] == 'None' or par['outEKFileName'] == 'none':
                        self.outEKFileName = trainName 
                        self.originalOutEKFileName = trainName
                    else:
                        self.outEKFileName = par['outEKFileName']+trainName          #Saved as text
                        self.originalOutEKFileName = par['outEKFileName']+trainName
                except:
                    self.outEKFileName = trainName 
                    self.originalOutEKFileName = trainName
                    
        try:
            self.filterAlgorithm = par['filterAlgorithm']                       #Saved as text
        except: #Default
            self.filterAlgorithm = 'multisurf'
        try:
            self.turfPercent = float(par['turfPercent'])                       #Saved as Boolean
        except: #Default
            self.turfPercent=0.05                              

        if self.filterAlgorithm == 'relieff':
            try:
                self.reliefNeighbors = int(par['reliefNeighbors'])                  #Saved as int
            except: #Default
                self.reliefNeighbors = 10
        if self.filterAlgorithm != 'multisurf': 
            try:
                self.reliefSampleFraction = float(par['reliefSampleFraction'])      #Saved as float
            except: #Default
                self.reliefSampleFraction = 1.0 
        try:
            self.onlyEKScores = bool(int(par['onlyEKScores']))                  #Saved as Boolean
        except: #Default
            self.onlyEKScores = False
            
        #Rule Compaction Parameters--------------------------------------------------------------------------------
        try:
            self.doRuleCompaction = bool(int(par['doRuleCompaction']))          #Saved as Boolean
        except: #Default
            self.doRuleCompaction = True
        try:
            self.onlyRC = bool(int(par['onlyRC']))                              #Saved as Boolean
        except: #Default
            self.onlyRC = False
        try:
            self.ruleCompactionMethod = par['ruleCompactionMethod']             #Saved as text
        except: #Default
            self.ruleCompactionMethod = 'QRF'
        #Population Reboot Parameters------------------------------------------------------------------
        try:
            self.doPopulationReboot = bool(int(par['doPopulationReboot']))      #Saved as Boolean
        except: #Default
            self.doPopulationReboot = False
        if self.doPopulationReboot:
            try:
                self.popRebootPath = self.outFileName+'_'+par['popRebootIteration']                           #Saved as text
            except:
                print('Constants: Error - Default value not available for "popRebootPath", please specify value in the configuration file.')
    
        self.firstEpochComplete = False
        
        #CALLBACKS - GUI
        self.epochCallbacks = []
        self.iterationCallbacks = []
        self.checkpointCallbacks = []
        
        #CONTROL OBJECTS - GUI
        self.stop = False
        self.forceCheckpoint = False
        
        if self.internalCrossValidation == 0 or self.internalCrossValidation == 1: 
            pass
        else: #Do internal CV
            self.originalTrainFile = copy.deepcopy(self.trainFile)
            self.originalTestFile = copy.deepcopy(self.testFile) 
        
        
    def referenceTimer(self, timer):
        """ Store reference to the timer object. """
        self.timer = timer
        
        
    def referenceEnv(self, e):
        """ Store reference to environment object. """
        self.env = e


    def referenceAttributeTracking(self, AT):
        """ Store reference to attribute tracking object. """
        self.AT = AT


    def referenceExpertKnowledge(self, EK):
        """ Store reference to attribute tracking object. """
        self.EK = EK
    
    
    def parseIterations(self):
        """ Format other key run parameters (i.e. maximum iterations, full evaluation checkpoints, and local evaluation tracking frequency. """
        checkpoints = self.learningIterations.split('.') #Parse the string specifying evaluation checkpoints, and the maximum number of learning iterations.
        for i in range(len(checkpoints)): #Convert checkpoint iterations from strings to ints.
            checkpoints[i] = int(checkpoints[i])
        self.learningCheckpoints = checkpoints
        self.maxLearningIterations = self.learningCheckpoints[(len(self.learningCheckpoints)-1)] 
        if self.trackingFrequency == 0:
            self.trackingFrequency = self.env.formatData.numTrainInstances  #Adjust tracking frequency to match the training data size - learning tracking occurs once every epoch


    def updateFileNames(self, part):
        """ A naming update method used when internal cross validation is applied. """
        tempName = copy.deepcopy(self.originalTrainFile)
        folderName = self.originalTrainFile#[0:len(self.originalTrainFile)-4]
        fileName = tempName.split('\\')
        fileName = fileName[len(fileName)-1]
        #fileName = fileName[0:len(self.originalTrainFile)-4]
        
        self.trainFile = folderName+'\\'+fileName+'_CV_'+str(part)+'_Train'
        self.testFile = folderName+'\\'+fileName+'_CV_'+str(part)+'_Test'
        self.outFileName = self.originalOutFileName+'_CV_'+str(part)+'_ExSTraCS'
        self.outEKFileName = self.originalOutEKFileName+'_CV_'+str(part)+'_ExSTraCS'
            
            
    def overrideParameters(self):
        """ Overrides user specified parameters for algorithm features that can not be applied to online datasets. """
        self.doAttributeTracking = False    #Saved as Boolean
        self.doAttributeFeedback = False    #Saved as Boolean
        self.useExpertKnowledge = False     #Saved as Boolean
        self.internal_EK_Generation = False #Saved as Boolean
        self.testFile = 'None'
        self.trainFile = 'None'
        self.doRuleCompaction = False       #Saved as Boolean
        self.onlyRC = False                 #Saved as Boolean
        if self.trackingFrequency == 0:
            self.trackingFrequency = 50
    
cons = Constants() #To access one of the above constant values from another module, import GHCS_Constants * and use "cons.Xconstant"
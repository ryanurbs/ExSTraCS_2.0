"""
Name:        ExSTraCS_Algorithm.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: The major controlling module of ExSTraCS.  Includes the major run loop which controls learning over a specified number of iterations.  Also includes
             periodic tracking of estimated performance, and checkpoints where complete evaluations of the ExSTraCS rule population is performed.
             
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
from exstracs_classifierset import ClassifierSet
from exstracs_prediction import *
from exstracs_at import *
from exstracs_rc import RuleCompaction
from exstracs_classaccuracy import ClassAccuracy
from exstracs_output import OutputFileManager
import copy
import random
import math
#------------------------------------------------------

class ExSTraCS:
    def __init__(self):
        """ Initializes the ExSTraCS algorithm """
        print("ExSTraCS: Initializing Algorithm...")
        #Global Parameters-------------------------------------------------------------------------------------
        self.population = None
        self.learnTrackOut = None  #Output file that stores tracking information during learning
        #-------------------------------------------------------
        # POPULATION REBOOT - Begin ExSTraCS learning from an existing saved rule population
        #-------------------------------------------------------
        if cons.doPopulationReboot: #If we are restarting from a previously saved rule population.
            try: #Re-open track learning file for continued tracking of progress.
                self.learnTrackOut = open(cons.outFileName+'_LearnTrack.txt','a')    
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                print('cannot open', cons.outFileName+'_LearnTrack.txt')
                raise 
            self.populationReboot()
            
        #-------------------------------------------------------
        # NORMAL ExSTraCS - Run ExSTraCS from scratch on given data
        #-------------------------------------------------------
        else:
            try: #Establish output file to store learning progress.
                self.learnTrackOut = open(cons.outFileName+'_LearnTrack.txt','w')     
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                print('cannot open', cons.outFileName+'_LearnTrack.txt')
                raise 
            else:
                self.learnTrackOut.write("Explore_Iteration\tMacroPopSize\tMicroPopSize\tAccuracy_Estimate\tAveGenerality\tExpRules\tTime(min)\n")

            # Instantiate Population---------
            self.population = ClassifierSet()
            self.exploreIter = 0
            self.correct  = [0.0 for i in range(cons.trackingFrequency)]
            self.predictionList = [] #For outputting raw testing predictions
            self.realList = []
            self.predictionSets = []

    def runExSTraCS(self):
        """ Runs the initialized ExSTraCS algorithm. """
        print("Beginning ExSTraCS learning iterations.")
        print("------------------------------------------------------------------------------------------------------------------------------------------------------")
        #-------------------------------------------------------
        # MAJOR LEARNING LOOP
        #-------------------------------------------------------
        while self.exploreIter < cons.maxLearningIterations and not cons.stop: #Major Learning Loop
            #-------------------------------------------------------
            # GET NEW INSTANCE AND RUN A LEARNING ITERATION
            #-------------------------------------------------------
            state_phenotype = cons.env.getTrainInstance() 
            self.runIteration(state_phenotype, self.exploreIter)

            #-------------------------------------------------------------------------------------------------------------------------------
            # EVALUATIONS OF ALGORITHM
            #-------------------------------------------------------------------------------------------------------------------------------
            cons.timer.startTimeEvaluation()
            #-------------------------------------------------------
            # TRACK LEARNING ESTIMATES
            #-------------------------------------------------------
            #Learning Tracking----------------------------------------------------------------------------------------------------------------------------------------
            if (self.exploreIter%cons.trackingFrequency) == (cons.trackingFrequency - 1) and self.exploreIter > 0:
                self.population.runPopAveEval(self.exploreIter) 
                trackedAccuracy = sum(self.correct)/float(cons.trackingFrequency) #Accuracy over the last "trackingFrequency" number of iterations.
                self.learnTrackOut.write(self.population.getPopTrack(trackedAccuracy, self.exploreIter+1,cons.trackingFrequency)) #Report learning progress to standard out and tracking file.
                for observer in cons.epochCallbacks:
                    observer(self.exploreIter, self.population, trackedAccuracy)
            cons.timer.stopTimeEvaluation()
            #-------------------------------------------------------
            # CHECKPOINT - COMPLETE EVALUTATION OF POPULATION - Evaluation strategy different for discrete vs continuous phenotypes
            #-------------------------------------------------------
            if (self.exploreIter + 1) in cons.learningCheckpoints or cons.forceCheckpoint:
                if(cons.forceCheckpoint):
                    cons.forceCheckpoint = False
                cons.timer.startTimeEvaluation()
                print("------------------------------------------------------------------------------------------------------------------------------------------------------")
                print("Running Population Evaluation after " + str(self.exploreIter + 1)+ " iterations.")
                self.population.runPopAveEval(self.exploreIter)
                self.population.runAttGeneralitySum()
                cons.env.startEvaluationMode()
                if cons.testFile != 'None': #If a testing file is available.
                    if cons.env.formatData.discretePhenotype: 
                        trainEval = self.doPopEvaluation(True)
                        testEval = self.doPopEvaluation(False)
                    else: 
                        print("Algorithm - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
                elif cons.trainFile != 'None':
                    if cons.env.formatData.discretePhenotype: 
                        trainEval = self.doPopEvaluation(True)
                        testEval = None
                    else: 
                        print("Algorithm - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
                else: #Online Environment
                    trainEval = None
                    testEval = None
                cons.env.stopEvaluationMode() #Returns to learning position in training data
                cons.timer.stopTimeEvaluation()
                #-----------------------------------------------------------------------------------------------------------------------------------------
                # WRITE OUTPUT FILES
                #-----------------------------------------------------------------------------------------------------------------------------------------
                cons.timer.startTimeOutFile()
                OutputFileManager().writePopStats(cons.outFileName, trainEval, testEval, self.exploreIter + 1, self.population, self.correct)
                OutputFileManager().writePop(cons.outFileName, self.exploreIter + 1, self.population)
                OutputFileManager().attCo_Occurence(cons.outFileName, self.exploreIter + 1, self.population)
                OutputFileManager().save_tracking(self.exploreIter, cons.outFileName)
                OutputFileManager().writePredictions(self.exploreIter, cons.outFileName, self.predictionList, self.realList, self.predictionSets)
                cons.timer.stopTimeOutFile()

                #GUI ONLY--------------------------------
                for observer in cons.checkpointCallbacks:
                    observer(trainEval, testEval)
                #----------------------------------------
                print("Continue Learning...")
                print("------------------------------------------------------------------------------------------------------------------------------------------------------")
                #-----------------------------------------------------------------------------------------------------------------------------------------
                # RULE COMPACTION
                #-----------------------------------------------------------------------------------------------------------------------------------------
                if self.exploreIter + 1 == cons.maxLearningIterations and cons.doRuleCompaction:
                    cons.timer.startTimeRuleCmp()
                    if testEval == None:
                        RuleCompaction(self.population, trainEval[0], None)
                    else:
                        RuleCompaction(self.population, trainEval[0], testEval[0])   
                    cons.timer.stopTimeRuleCmp() 
                    #-----------------------------------------------------------------------------------------------------------------------------------------
                    # GLOBAL EVALUATION OF COMPACTED RULE POPULATION
                    #-----------------------------------------------------------------------------------------------------------------------------------------
                    cons.timer.startTimeEvaluation()
                    self.population.recalculateNumerositySum()
                    self.population.runPopAveEval(self.exploreIter)
                    self.population.runAttGeneralitySum()
                    #----------------------------------------------------------
                    cons.env.startEvaluationMode()
                    if cons.testFile != 'None': #If a testing file is available.
                        if cons.env.formatData.discretePhenotype:
                            trainEval = self.doPopEvaluation(True)
                            testEval = self.doPopEvaluation(False)
                        else: 
                            print("Algorithm - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
                    else:
                        if cons.env.formatData.discretePhenotype: 
                            trainEval = self.doPopEvaluation(True)
                            testEval = None
                        else: 
                            print("Algorithm - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
                    cons.env.stopEvaluationMode()
                    cons.timer.stopTimeEvaluation()

                    #-----------------------------------------------------------------------------------------------------------------------------------------
                    # WRITE OUTPUT FILES
                    #-----------------------------------------------------------------------------------------------------------------------------------------
                    cons.timer.startTimeOutFile()
                    OutputFileManager().writePopStats(cons.outFileName+"_RC_"+cons.ruleCompactionMethod, trainEval, testEval, self.exploreIter + 1, self.population, self.correct)
                    OutputFileManager().writePop(cons.outFileName+"_RC_"+cons.ruleCompactionMethod, self.exploreIter + 1, self.population)
                    OutputFileManager().attCo_Occurence(cons.outFileName+"_RC_"+cons.ruleCompactionMethod, self.exploreIter + 1, self.population)
                    OutputFileManager().writePredictions(self.exploreIter, cons.outFileName+"_RC_"+cons.ruleCompactionMethod, self.predictionList, self.realList, self.predictionSets)
                    cons.timer.stopTimeOutFile()
                    
            #GUI ONLY--------------------------------
            for observer in cons.iterationCallbacks:
                observer()
            
            #-------------------------------------------------------
            # ADJUST MAJOR VALUES FOR NEXT ITERATION
            #-------------------------------------------------------
            self.exploreIter += 1 
            cons.env.newInstance(True) #move to next instance in training set
            
        # Once ExSTraCS has reached the last learning iteration, close the tracking file 
        self.learnTrackOut.close() 
        print("ExSTraCS Run Complete")
        
        
    def runIteration(self, state_phenotype, exploreIter):
        """ Run single ExSTraCS learning iteration. """
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # FORM A MATCH SET - includes covering
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.makeMatchSet(state_phenotype, exploreIter)
        cons.timer.startTimeEvaluation()
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # MAKE A PREDICTION - Utilized here for tracking estimated learning progress.  Typically used in the explore phase of many LCS algorithms.
        #-----------------------------------------------------------------------------------------------------------------------------------------
        prediction = Prediction(self.population)
        phenotypePrediction = prediction.getDecision()  
        #-------------------------------------------------------
        # PREDICTION NOT POSSIBLE
        #-------------------------------------------------------
        if phenotypePrediction == None or phenotypePrediction == 'Tie': 
            if cons.env.formatData.discretePhenotype:
                phenotypePrediction = random.choice(cons.env.formatData.phenotypeList)
            else:
                phenotypePrediction = random.randrange(cons.env.formatData.phenotypeList[0],cons.env.formatData.phenotypeList[1],(cons.env.formatData.phenotypeList[1]-cons.env.formatData.phenotypeList[0])/float(1000))
        else: 
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE PREDICTION
        #-------------------------------------------------------
            if cons.env.formatData.discretePhenotype:
                if phenotypePrediction == state_phenotype[1]:
                    self.correct[exploreIter%cons.trackingFrequency] = 1
                else:
                    self.correct[exploreIter%cons.trackingFrequency] = 0
            else:
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE PREDICTION
        #-------------------------------------------------------
                predictionError = math.fabs(phenotypePrediction - float(state_phenotype[1]))
                phenotypeRange = cons.env.formatData.phenotypeList[1] - cons.env.formatData.phenotypeList[0]
                accuracyEstimate = 1.0 - (predictionError / float(phenotypeRange))
                self.correct[exploreIter%cons.trackingFrequency] = accuracyEstimate
        cons.timer.stopTimeEvaluation()
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # FORM A CORRECT SET
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.makeCorrectSet(state_phenotype[1])
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # UPDATE PARAMETERS
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.updateSets(exploreIter)
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # SUBSUMPTION - APPLIED TO CORRECT SET - A heuristic for addition additional generalization pressure to ExSTraCS
        #-----------------------------------------------------------------------------------------------------------------------------------------
        if cons.doSubsumption:
            cons.timer.startTimeSubsumption()
            self.population.doCorrectSetSubsumption()
            cons.timer.stopTimeSubsumption()
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # ATTRIBUTE TRACKING AND FEEDBACK - A long-term memory mechanism tracked for each instance in the dataset and used to help guide the GA
        #-----------------------------------------------------------------------------------------------------------------------------------------
        if cons.doAttributeTracking:
            cons.timer.startTimeAT()
            cons.AT.updateAttTrack(self.population) 
            if cons.doAttributeFeedback:
                cons.AT.updatePercent(exploreIter)
                cons.AT.genTrackProb() 
            cons.timer.stopTimeAT()
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # RUN THE GENETIC ALGORITHM - Discover new offspring rules from a selected pair of parents
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.runGA(exploreIter, state_phenotype[0], state_phenotype[1]) #GA is run within the correct set. 
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # SELECT RULES FOR DELETION - This is done whenever there are more rules in the population than 'N', the maximum population size.
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.deletion(exploreIter)
        self.population.clearSets() #Clears the match and correct sets for the next learning iteration
        
        
    def doPopEvaluation(self, isTrain):
        """ Performs a complete evaluation of the current rule population.  Discrete phenotype only.  The population is unchanged throughout this evaluation. Works on both training and testing data. """
        if isTrain:
            myType = "TRAINING"
        else:
            myType = "TESTING"
        noMatch = 0                     # How often does the population fail to have a classifier that matches an instance in the data.
        tie = 0                         # How often can the algorithm not make a decision between classes due to a tie.
        cons.env.resetDataRef(isTrain)  # Go to the first instance in dataset
        phenotypeList = cons.env.formatData.phenotypeList 
        #Initialize dictionary entry for each class----
        classAccDict = {}
        for each in phenotypeList:
            classAccDict[each] = ClassAccuracy()
        #----------------------------------------------
        if isTrain:
            instances = cons.env.formatData.numTrainInstances
        else:
            instances = cons.env.formatData.numTestInstances
        self.predictionList = []
        self.predictionSets = []
        self.realList = []
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # GET PREDICTION AND DETERMINE PREDICTION STATUS
        #-----------------------------------------------------------------------------------------------------------------------------------------
        for inst in range(instances):
            if isTrain:
                state_phenotype = cons.env.getTrainInstance()
            else:
                state_phenotype = cons.env.getTestInstance()
            #-----------------------------------------------------------------------------
            self.population.makeEvalMatchSet(state_phenotype[0])
            prediction = Prediction(self.population)
            phenotypeSelection = prediction.getDecision() 
            if not isTrain:
                phenotypeSet = prediction.getSet()
                self.predictionList.append(phenotypeSelection) #Used to output raw test predictions.
                self.predictionSets.append(phenotypeSet)
                self.realList.append(state_phenotype[1])
            #-----------------------------------------------------------------------------
            if phenotypeSelection == None: 
                noMatch += 1
            elif phenotypeSelection == 'Tie':
                tie += 1
            else: #Instances which failed to be covered are excluded from the initial accuracy calculation 
                for each in phenotypeList:
                    thisIsMe = False
                    accuratePhenotype = False
                    truePhenotype = state_phenotype[1]
                    if each == truePhenotype:
                        thisIsMe = True #Is the current phenotype the true data phenotype.
                    if phenotypeSelection == truePhenotype:
                        accuratePhenotype = True
                    classAccDict[each].updateAccuracy(thisIsMe, accuratePhenotype)
                        
            cons.env.newInstance(isTrain) #next instance
            self.population.clearSets() 
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # CALCULATE ACCURACY - UNLIKELY SITUATION WHERE NO MATCHING RULES FOUND - In either Training or Testing data (this can happen in testing data when strong training overfitting occurred)
        #-----------------------------------------------------------------------------------------------------------------------------------------
        if noMatch == instances:
            randomProb = float (1.0 / len(cons.env.formatData.phenotypeList))
            print("-----------------------------------------------")
            print(str(myType)+" Accuracy Results:-------------")
            print("Instance Coverage = "+ str(0)+ '%')
            print("Prediction Ties = "+ str(0)+ '%')
            print(str(0) + ' out of ' + str(instances) + ' instances covered and correctly classified.')
            print("Standard Accuracy (Adjusted) = " + str(randomProb))
            print("Balanced Accuracy (Adjusted) = " + str(randomProb))
            #Balanced and Standard Accuracies will only be the same when there are equal instances representative of each phenotype AND there is 100% covering. (NOTE even at 100% covering, the values may differ due to subtle float calculation differences in the computer)
            resultList = [randomProb, 0]  
            return resultList
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # CALCULATE ACCURACY
        #-----------------------------------------------------------------------------------------------------------------------------------------
        else:
            #----------------------------------------------------------------------------------------------
            #Calculate Standard Accuracy------------------------------------
            standardAccuracy = 0
            for each in phenotypeList: 
                instancesCorrectlyClassified = classAccDict[each].T_myClass + classAccDict[each].T_otherClass
                instancesIncorrectlyClassified = classAccDict[each].F_myClass + classAccDict[each].F_otherClass 
                classAccuracy = float(instancesCorrectlyClassified) / float(instancesCorrectlyClassified + instancesIncorrectlyClassified)
                standardAccuracy += classAccuracy
            standardAccuracy = standardAccuracy / float(len(phenotypeList))

            #Calculate Balanced Accuracy---------------------------------------------
            balancedAccuracy = 0
            for each in phenotypeList: 
                try:
                    sensitivity = classAccDict[each].T_myClass / (float(classAccDict[each].T_myClass + classAccDict[each].F_otherClass))
                except:
                    sensitivity = 0.0
                try:
                    specificity = classAccDict[each].T_otherClass / (float(classAccDict[each].T_otherClass + classAccDict[each].F_myClass))
                except:
                    specificity = 0.0
                
                balancedClassAccuracy = (sensitivity + specificity) / 2.0
                balancedAccuracy += balancedClassAccuracy
                
            balancedAccuracy = balancedAccuracy / float(len(phenotypeList))  
    
            #Adjustment for uncovered instances - to avoid positive or negative bias we incorporate the probability of guessing a phenotype by chance (e.g. 50% if two phenotypes)---------------------------------------
            predictionFail = float(noMatch)/float(instances)
            predictionTies = float(tie)/float(instances)
            instanceCoverage = 1.0 - predictionFail
            predictionMade = 1.0 - (predictionFail + predictionTies)
            
            adjustedStandardAccuracy = (standardAccuracy * predictionMade) + ((1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))
            adjustedBalancedAccuracy = (balancedAccuracy * predictionMade) + ((1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))
            
            #Adjusted Balanced Accuracy is calculated such that instances that did not match have a consistent probability of being correctly classified in the reported accuracy.
            print("-----------------------------------------------")
            print(str(myType)+" Accuracy Results:-------------")
            print("Instance Coverage = "+ str(instanceCoverage*100.0)+ '%')
            print("Prediction Ties = "+ str(predictionTies*100.0)+ '%')
            print(str(instancesCorrectlyClassified) + ' out of ' + str(instances) + ' instances covered and correctly classified.')
            print("Standard Accuracy (Adjusted) = " + str(adjustedStandardAccuracy))
            print("Balanced Accuracy (Adjusted) = " + str(adjustedBalancedAccuracy))
            #Balanced and Standard Accuracies will only be the same when there are equal instances representative of each phenotype AND there is 100% covering. (NOTE even at 100% covering, the values may differ due to subtle float calculation differences in the computer)
            resultList = [adjustedBalancedAccuracy, instanceCoverage]  
            return resultList
        
    
    def populationReboot(self):
        """ Manages the loading and continued learning/evolution of a previously saved ExSTraCS classifier population. """
        cons.timer.setTimerRestart(cons.popRebootPath) #Rebuild timer objects
        
        #Extract last iteration from file name---------------------------------------------
        temp = cons.popRebootPath.split('_')
        iterRef = len(temp)-1
        completedIterations = int(temp[iterRef])
        print("Rebooting rule population after " +str(completedIterations)+ " iterations.")
        self.exploreIter = completedIterations-1
        for i in range(len(cons.learningCheckpoints)):
            cons.learningCheckpoints[i] += completedIterations
        cons.maxLearningIterations += completedIterations

        #Rebuild existing population from text file.--------
        self.population = ClassifierSet(cons.popRebootPath)
        #---------------------------------------------------
        try: #Obtain correct track
            f = open(cons.popRebootPath+"_PopStats.txt", 'rU')
            correctRef = 39 #Reference for tracking learning accuracy estimate stored in PopStats.
            tempLine = None
            for i in range(correctRef):
                tempLine = f.readline()
            tempList = tempLine.strip().split('\t')
            self.correct = tempList
            if cons.env.formatData.discretePhenotype:
                for i in range(len(self.correct)):
                    self.correct[i] = int(self.correct[i])
            else:
                for i in range(len(self.correct)):
                    self.correct[i] = float(self.correct[i])
            f.close()
        except IOError as xxx_todo_changeme3:
            (errno, strerror) = xxx_todo_changeme3.args
            print(("I/O error(%s): %s" % (errno, strerror)))
            raise
        

    def runRConly(self):
        """ Run rule compaction on an existing rule population. """
        print("Initializing Rule Compaction...")
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # CHECK FOR POPULATION REBOOT - Required for running Rule Compaction only on an existing saved rule population.
        #-----------------------------------------------------------------------------------------------------------------------------------------
        if not cons.doPopulationReboot:  
            print("Algorithm: Error - Existing population required to run rule compaction alone.")
            return
        
        try:
            fileObject = open(cons.popRebootPath+"_PopStats.txt", 'rU')  # opens each datafile to read.
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', cons.popRebootPath+"_PopStats.txt")
            raise 

            
        #Retrieve last training and testing accuracies from saved file---------
        tempLine = None
        for i in range(3):
            tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')
        trainAcc = float(tempList[0])
        if cons.testFile != 'None': #If a testing file is available.
            testAcc = float(tempList[1])
        else:
            testAcc = None
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # RULE COMPACTION
        #-----------------------------------------------------------------------------------------------------------------------------------------
        cons.timer.startTimeRuleCmp()
        RuleCompaction(self.population, trainAcc, testAcc)  
        cons.timer.stopTimeRuleCmp() 
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # GLOBAL EVALUATION OF COMPACTED RULE POPULATION
        #-----------------------------------------------------------------------------------------------------------------------------------------
        cons.timer.startTimeEvaluation()
        self.population.recalculateNumerositySum()
        self.population.runPopAveEval(self.exploreIter)
        self.population.runAttGeneralitySum()
        #----------------------------------------------------------
        cons.env.startEvaluationMode()
        if cons.testFile != 'None': #If a testing file is available.
            if cons.env.formatData.discretePhenotype: 
                trainEval = self.doPopEvaluation(True)
                testEval = self.doPopEvaluation(False)
            else: 
                print("Algorithm - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
        elif cons.trainFile != 'None':
            if cons.env.formatData.discretePhenotype: 
                trainEval = self.doPopEvaluation(True)
                testEval = None
            else: 
                print("Algorithm - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
        else: #Online Environment
            trainEval = None
            testEval = None
        
        cons.env.stopEvaluationMode()
        cons.timer.stopTimeEvaluation()
        #------------------------------------------------------------------------------
        cons.timer.returnGlobalTimer() 
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # WRITE OUTPUT FILES
        #-----------------------------------------------------------------------------------------------------------------------------------------
        OutputFileManager().writePopStats(cons.outFileName+"_RC_"+cons.ruleCompactionMethod, trainEval, testEval, self.exploreIter + 1, self.population, self.correct)
        OutputFileManager().writePop(cons.outFileName+"_RC_"+cons.ruleCompactionMethod, self.exploreIter + 1, self.population)
        OutputFileManager().attCo_Occurence(cons.outFileName+"_RC_"+cons.ruleCompactionMethod, self.exploreIter + 1, self.population)
        OutputFileManager().writePredictions(self.exploreIter, cons.outFileName+"_RC_"+cons.ruleCompactionMethod, self.predictionList, self.realList, self.predictionSets)
        #------------------------------------------------------------------------------------------------------------ 
        print("Rule Compaction Complete")
        

    def runTestonly(self):
        """ Run testing dataset evaluation on an existing rule population. """
        print("Initializing Evaluation of Testing Dataset...")
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # CHECK FOR POPULATION REBOOT - Required for running Testing Evaluation only on an existing saved rule population.
        #-----------------------------------------------------------------------------------------------------------------------------------------
        if not cons.doPopulationReboot:  
            print("Algorithm: Error - Existing population required to run rule compaction alone.")
            return

        #----------------------------------------------------------
        cons.env.startEvaluationMode()
        if cons.testFile != 'None': #If a testing file is available.
            if cons.env.formatData.discretePhenotype: 
                testEval = self.doPopEvaluation(False)
            else: 
                print("Algorithm - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
        else: #Online Environment
            testEval = None
                    
        cons.env.stopEvaluationMode()
        cons.timer.stopTimeEvaluation()
        #------------------------------------------------------------------------------
        cons.timer.returnGlobalTimer() 
        OutputFileManager().editPopStats(testEval)
        OutputFileManager().writePredictions(self.exploreIter, cons.outFileName, self.predictionList, self.realList, self.predictionSets)
        print("Testing Evaluation Complete")
        
"""
Name:        ExSTraCS_Rule Compaction.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: Includes several rule compaction/rule filter strategies, which can be selected as a post-processing stage following ExSTraCS classifier 
             population learning. Fu1, Fu2, and CRA2 were previously proposed/published strategies from other authors.  QRC, PDRC, and QRF were 
             proposed and published by Jie Tan, Jason Moore, and Ryan Urbanowicz in "Rapid Rule Compaction Strategies for Global Knowledge Discovery
             in a Supervised Learning Classifier System." [2013].
             
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
from exstracs_classaccuracy import ClassAccuracy
from exstracs_prediction import *
from exstracs_classifierset import ClassifierSet
import copy
import math
#------------------------------------------------------

class RuleCompaction:
    def __init__(self, pop, originalTrainAcc, originalTestAcc):
        """ Initialize and run the specified rule compaction strategy. """
        print("---------------------------------------------------------------------------------------------------------")
        print("Starting Rule Compaction Algorithm ("+str(cons.ruleCompactionMethod)+") ...")
        self.pop = pop
        self.originalTrainAcc = originalTrainAcc
        self.originalTestAcc = originalTestAcc
        
        #Outside Rule Compaction Strategies------------------------------------------------------------------------------------
        if cons.ruleCompactionMethod == 'Fu1': #(Implemented by Jie Tan, Referred to as 'A12' in original code.)
            self.Approach_Fu1()  
        elif cons.ruleCompactionMethod =='Fu2': #(Implemented by Jie Tan, Referred to as 'A13' in original code.)
            self.Approach_Fu2()  
        elif cons.ruleCompactionMethod =='CRA2': #(Implemented by Jie Tan, Referred to as 'A8' or 'Dixon' in original code.)
            self.Approach_CRA2() 
        #------------------------------------------------------------------------------------------------------------------------
        #ExSTraCS Original Rule Compaction Strategies:--------------------------------------------------------------------------------
        elif cons.ruleCompactionMethod =='QRC': #Quick Rule Compaction (Developed by Jie Tan, Referred to as 'A9' or 'UCRA' in original code.)
            self.Approach_QRC() 
        elif cons.ruleCompactionMethod =='PDRC': #Parameter Driven Rule Compaction - (Developed by Jie Tan, Referred to as 'A17' or 'QCRA' in original code.)
            self.Approach_PDRC()  
        elif cons.ruleCompactionMethod == 'QRF': #Quick Rule Filter - (Developed by Ryan Urbanowicz, Referred to as 'Quick Rule Cleanup' or 'QRC' in original code.) 
            self.Approach_QRF()
        else:
            print("RuleCompaction: Error - specified rule compaction strategy not found.")
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # COMPACTION STRATEGIES
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def Approach_Fu1(self):
        """ This approach completely follows Fu's first approach. In the third stage, the number of instances a rule matched is used to rank 
        the rules and guide covering. Ranking list is updated each time some instances are covered and removed from the training set. """
        
        #Order Classifier Set---------------------------------------------------------------------------------------------------------
        lastGood_popSet = sorted(self.pop.popSet, key = self.numerositySort)
        self.pop.popSet = lastGood_popSet[:]
        print("Starting number of classifiers = " + str(len(self.pop.popSet))) 
        print("Original Training Accuracy = " +str(self.originalTrainAcc))
        print("Original Testing Accuracy = " +str(self.originalTestAcc))
        
        #STAGE 1----------------------------------------------------------------------------------------------------------------------
        keepGoing = True
        while keepGoing:
            del self.pop.popSet[0] #Remove next classifier
            newAccuracy = self.performanceEvaluation(True) #Perform classifier set training accuracy evaluation

            if newAccuracy < self.originalTrainAcc:
                keepGoing = False
                self.pop.popSet = lastGood_popSet[:]
            else:
                lastGood_popSet = self.pop.popSet[:]
            if len(self.pop.popSet) == 0:
                keepGoing = False
        print("STAGE 1 Ended: Classifiers Remaining = " +str(len(self.pop.popSet))) 
        
        #STAGE 2----------------------------------------------------------------------------------------------------------------------
        retainedClassifiers = []
        RefAccuracy = self.originalTrainAcc
        for i in range(len(self.pop.popSet)): 
            heldClassifier = self.pop.popSet[0]
            del self.pop.popSet[0]
            newAccuracy = self.performanceEvaluation(True) #Perform classifier set training accuracy evaluation

            if newAccuracy < RefAccuracy:
                retainedClassifiers.append(heldClassifier)
                RefAccuracy = newAccuracy

        self.pop.popSet = retainedClassifiers
        print("STAGE 2 Ended: Classifiers Remaining = " +str(len(self.pop.popSet))) 
        
        #STAGE 3----------------------------------------------------------------------------------------------------------------------
        finalClassifiers = []
        completelyGeneralRuleRef = None
        if len(self.pop.popSet) == 0: #Stop Check
            keepGoing = False
        else:
            keepGoing = True

        #Make the match count list in preparation for state 3------------------------------------------------------------------------- 
        matchCountList = [0.0 for v in range(len(self.pop.popSet))] 
        cons.env.startEvaluationMode()
        for i in range(len(self.pop.popSet)): #For the population of classifiers
            cons.env.resetDataRef(True)
            for j in range(cons.env.formatData.numTrainInstances): #For each instance in training data
                cl = self.pop.popSet[i]
                state = cons.env.getTrainInstance()[0]
                doesMatch = cl.match(state)
                if doesMatch:
                    matchCountList[i] += 1
                cons.env.newInstance(True)
            if len(self.pop.popSet[i].condition) == 0:
                completelyGeneralRuleRef = i
      
        cons.env.stopEvaluationMode()
        if completelyGeneralRuleRef != None: #gets rid of completely general rule.
            del matchCountList[completelyGeneralRuleRef]
            del self.pop.popSet[completelyGeneralRuleRef]

        #----------------------------------------------------------------------------------------------------------------------------
        tempEnv = copy.deepcopy(cons.env)
        trainingData = tempEnv.formatData.trainFormatted 
        while len(trainingData) > 0 and keepGoing: 
            bestRef = None
            bestValue = None
            for i in range(len(matchCountList)):
                if bestValue == None or bestValue < matchCountList[i]:
                    bestRef = i
                    bestValue = matchCountList[i]
                    
            if bestValue == 0.0 or len(self.pop.popSet) < 1:
                keepGoing = False
                continue

            #Update Training Data----------------------------------------------------------------------------------------------------
            matchedData = 0
            w = 0
            cl = self.pop.popSet[bestRef]
            for i in range(len(trainingData)):
                state = trainingData[w][0]
                doesMatch = cl.match(state)
                if doesMatch:
                    matchedData += 1
                    del trainingData[w]
                else:
                    w += 1
            if matchedData > 0:
                finalClassifiers.append(self.pop.popSet[bestRef]) #Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!
            
            #Update classifier list
            del self.pop.popSet[bestRef]

            #re-calculate match count list
            matchCountList = [0.0 for v in range(len(self.pop.popSet))]
            for i in range(len(self.pop.popSet)):
                dataRef = 0 
                for j in range(len(trainingData)): #For each instance in training data
                    cl = self.pop.popSet[i]
                    state = trainingData[dataRef][0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchCountList[i] += 1
                    dataRef +=1
                
            if len(self.pop.popSet) == 0:
                keepGoing = False
           
        self.pop.popSet = finalClassifiers 
        print("STAGE 3 Ended: Classifiers Remaining = " +str(len(self.pop.popSet))) 
        
        
    ############################################################################################################################################################################################
    def Approach_Fu2(self):
        """ This approach completely follows Fu's second approach. All three stages use accuracy to sort rules."""
        #Order Classifier Set---------------------------------------------------------------------------------------------------------
        lastGood_popSet = sorted(self.pop.popSet, key = self.numerositySort)
        self.pop.popSet = lastGood_popSet[:]
        print("Starting number of classifiers = " + str(len(self.pop.popSet))) 
        print("Original Training Accuracy = " +str(self.originalTrainAcc))
        print("Original Testing Accuracy = " +str(self.originalTestAcc))
        
        #STAGE 1----------------------------------------------------------------------------------------------------------------------
        keepGoing = True
        while keepGoing:
            del self.pop.popSet[0] #Remove next classifier
            newAccuracy = self.performanceEvaluation(True) #Perform classifier set training accuracy evaluation
            if newAccuracy < self.originalTrainAcc:
                keepGoing = False
                self.pop.popSet = lastGood_popSet[:]
            else:
                lastGood_popSet = self.pop.popSet[:]
            if len(self.pop.popSet) == 0:
                keepGoing = False
        print("STAGE 1 Ended: Classifiers Remaining = " +str(len(self.pop.popSet))) 
        
        #STAGE 2----------------------------------------------------------------------------------------------------------------------
        retainedClassifiers = []
        RefAccuracy = self.originalTrainAcc
        for i in range(len(self.pop.popSet)): 
            heldClassifier = self.pop.popSet[0]
            del self.pop.popSet[0]
            newAccuracy = self.performanceEvaluation(True) #Perform classifier set training accuracy evaluation
            
            if newAccuracy < RefAccuracy:
                retainedClassifiers.append(heldClassifier)
                RefAccuracy = newAccuracy
                
        self.pop.popSet = retainedClassifiers
        print("STAGE 2 Ended: Classifiers Remaining = " +str(len(self.pop.popSet))) 
        
        #STAGE 3----------------------------------------------------------------------------------------------------------------------
        Sort_popSet = sorted(self.pop.popSet, key = self.numerositySort, reverse = True)
        self.pop.popSet = Sort_popSet[:]
        RefAccuracy = self.performanceEvaluation(True)
        
        if len(self.pop.popSet) == 0: #Stop check
            keepGoing = False
        else:
            keepGoing = True
        
        for i in range(len(self.pop.popSet)): 
            heldClassifier = self.pop.popSet[0]
            del self.pop.popSet[0]
            newAccuracy = self.performanceEvaluation(True) #Perform classifier set training accuracy evaluation
            
            if newAccuracy < RefAccuracy:
                self.pop.popSet.append(heldClassifier)
            else:
                RefAccuracy = newAccuracy

        print("STAGE 3 Ended: Classifiers Remaining = " +str(len(self.pop.popSet))) 


    ############################################################################################################################################################################################
    def Approach_CRA2(self):
        """ This approach is based on Dixon's and Shoeleh's method. For each instance, form a match set and then a correct set. The most useful rule in 
            the correct set is moved into the final ruleset. In this approach, the most useful rule has the largest product of accuracy
            and generality."""    
    
        print("Starting number of classifiers = " + str(len(self.pop.popSet))) 
        print("Original Training Accuracy = " +str(self.originalTrainAcc))
        print("Original Testing Accuracy = " +str(self.originalTestAcc))
        
        retainedClassifiers = []
        self.matchSet = [] 
        self.correctSet = []
        
        cons.env.startEvaluationMode()
        cons.env.resetDataRef(True)     
        for j in range(cons.env.formatData.numTrainInstances):
            state_phenotype = cons.env.getTrainInstance()
            state = state_phenotype[0]
            phenotype = state_phenotype[1]
            
            #Create MatchSet
            for i in range(len(self.pop.popSet)):
                cl = self.pop.popSet[i]                                 
                if cl.match(state):                                
                    self.matchSet.append(i)
                    
            #Create CorrectSet
            if cons.env.formatData.discretePhenotype:
                for i in range(len(self.matchSet)):
                    ref = self.matchSet[i]
                    if self.pop.popSet[ref].phenotype == phenotype:
                        self.correctSet.append(ref)
            else:
                for i in range(len(self.matchSet)):
                    ref = self.matchSet[i]
                    if float(phenotype) <= float(self.pop.popSet[ref].phenotype[1]) and float(phenotype) >= float(self.pop.popSet[ref].phenotype[0]):
                        self.correctSet.append(ref)
        
            #Find the rule with highest accuracy, generality product
            highestValue = 0
            highestRef = 0
            for i in range(len(self.correctSet)):
                ref = self.correctSet[i]
                product = self.pop.popSet[ref].accuracy * (cons.env.formatData.numAttributes - len(self.pop.popSet[ref].condition)) / float(cons.env.formatData.numAttributes)
                if product > highestValue:
                    highestValue = product
                    highestRef = ref
            
            #If the rule is not already in the final ruleset, move it to the final ruleset
            if highestValue == 0 or self.pop.popSet[highestRef] in retainedClassifiers:
                pass
            else:
                retainedClassifiers.append(self.pop.popSet[highestRef])

            #Move to the next instance                
            cons.env.newInstance(True)
            self.matchSet = [] 
            self.correctSet = []
        cons.env.stopEvaluationMode()
        
        self.pop.popSet = retainedClassifiers
        print("STAGE 1 Ended: Classifiers Remaining = " +str(len(self.pop.popSet))) 
    
    
    ############################################################################################################################################################################################  
    def Approach_QRC(self):
        """Called QCRA in the paper. It uses fitness to rank rules and guide covering. It's the same as Approach 15, but the code is re-written in 
        order to speed up."""
        
        print("Starting number of classifiers = " + str(len(self.pop.popSet))) 
        print("Original Training Accuracy = " +str(self.originalTrainAcc))
        print("Original Testing Accuracy = " +str(self.originalTestAcc))
        
        #STAGE 1----------------------------------------------------------------------------------------------------------------------
        finalClassifiers = []
        if len(self.pop.popSet) == 0: #Stop check
            keepGoing = False
        else:
            keepGoing = True

        lastGood_popSet = sorted(self.pop.popSet, key = self.accuracySort, reverse = True)
        self.pop.popSet = lastGood_popSet[:]
        
        tempEnv = copy.deepcopy(cons.env)
        trainingData = tempEnv.formatData.trainFormatted
        
        while len(trainingData) > 0 and keepGoing: 
            newTrainSet = []
            matchedData = 0
            for w in range(len(trainingData)):
                cl = self.pop.popSet[0]
                state = trainingData[w][0]
                doesMatch = cl.match(state)
                if doesMatch:
                    matchedData += 1
                else:
                    newTrainSet.append(trainingData[w])
            if matchedData > 0:
                finalClassifiers.append(self.pop.popSet[0]) #Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!
            #Update classifier list and training set list
            trainingData = newTrainSet
            del self.pop.popSet[0]
            if len(self.pop.popSet) == 0:
                keepGoing = False
           
        self.pop.popSet = finalClassifiers 
        print("STAGE 1 Ended: Classifiers Remaining = " +str(len(self.pop.popSet))) 


    ############################################################################################################################################################################################
    def Approach_PDRC(self):
        """ This approach is based on Dixon's approach, called UCRA in the paper. For each instance, form a match set and then a correct set. 
        The most useful rule in the correct set is moved into the final ruleset. In this approach, the most useful rule has the largest 
        product of accuracy, numerosity and generality.""" 
        
        print("Starting number of classifiers = " + str(len(self.pop.popSet))) 
        print("Original Training Accuracy = " +str(self.originalTrainAcc))
        print("Original Testing Accuracy = " +str(self.originalTestAcc))
        
        
        retainedClassifiers = []
        self.matchSet = [] 
        self.correctSet = []
        
        cons.env.startEvaluationMode()
        cons.env.resetDataRef(True)     
        for j in range(cons.env.formatData.numTrainInstances):
            state_phenotype = cons.env.getTrainInstance()
            state = state_phenotype[0]
            phenotype = state_phenotype[1]
            
            #Create MatchSet
            for i in range(len(self.pop.popSet)):
                cl = self.pop.popSet[i]                                 
                if cl.match(state):                                
                    self.matchSet.append(i)
                    
            #Create CorrectSet
            if cons.env.formatData.discretePhenotype:
                for i in range(len(self.matchSet)):
                    ref = self.matchSet[i]
                    if self.pop.popSet[ref].phenotype == phenotype:
                        self.correctSet.append(ref)
            else:
                for i in range(len(self.matchSet)):
                    ref = self.matchSet[i]
                    if float(phenotype) <= float(self.pop.popSet[ref].phenotype[1]) and float(phenotype) >= float(self.pop.popSet[ref].phenotype[0]):
                        self.correctSet.append(ref)
            #Find the rule with highest accuracy, generality and numerosity product
            highestValue = 0
            highestRef = 0
            for i in range(len(self.correctSet)):
                ref = self.correctSet[i]
                product = self.pop.popSet[ref].accuracy * (cons.env.formatData.numAttributes - len(self.pop.popSet[ref].condition)) / float(cons.env.formatData.numAttributes) * self.pop.popSet[ref].numerosity
                if product > highestValue:
                    highestValue = product
                    highestRef = ref
        
            #If the rule is not already in the final ruleset, move it to the final ruleset
            if highestValue == 0 or self.pop.popSet[highestRef] in retainedClassifiers:
                pass
            else:
                retainedClassifiers.append(self.pop.popSet[highestRef])

            #Move to the next instance                
            cons.env.newInstance(True)
            self.matchSet = [] 
            self.correctSet = []
        cons.env.stopEvaluationMode()
        
        self.pop.popSet = retainedClassifiers
        print("STAGE 1 Ended: Classifiers Remaining = " +str(len(self.pop.popSet))) 
        
 
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # FILTER STRATEGIES
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def Approach_QRF(self):
        """ An extremely fast rule compaction strategy. Removes any rule with an accuracy below 50% and any rule that covers only one instance, but specifies more than one attribute
         (won't get rid of rare variant rules)"""
        
        print("Starting number of classifiers = " + str(len(self.pop.popSet))) 
        print("Original Training Accuracy = " +str(self.originalTrainAcc))
        print("Original Testing Accuracy = " +str(self.originalTestAcc))
        
        #STAGE 1----------------------------------------------------------------------------------------------------------------------
        retainedClassifiers = []
        for i in range(len(self.pop.popSet)): 
            if self.pop.popSet[i].accuracy <= 0.5 or (self.pop.popSet[i].correctCover == 1 and len(self.pop.popSet[i].specifiedAttList) > 1):
                pass
            else:
                retainedClassifiers.append(self.pop.popSet[i])
        self.pop.popSet = retainedClassifiers
        print("STAGE 1 Ended: Classifiers Remaining = " +str(len(self.pop.popSet))) 


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # EVALUTATION METHODS
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def performanceEvaluation(self, isTrain):
        """ Performs Training or Testing Evaluation"""
        if cons.env.formatData.discretePhenotype:
            adjustedBalancedAccuracy = self.doPopEvaluation(isTrain)
        else: #Continuous Phenotype
            print("RuleCompaction - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
            
        return adjustedBalancedAccuracy
    
    
    def doPopEvaluation(self, isTrain):
        """ Performs evaluation of population via the copied environment.  The population is maintained unchanging throughout the evaluation.
        Works on both training and testing data. """
        cons.env.startEvaluationMode()
        noMatch = 0 #How often does the population fail to have a classifier that matches an instance in the data.
        tie = 0 #How often can the algorithm not make a decision between classes due to a tie.
        cons.env.resetDataRef(isTrain) #Go to first instance in data set
        phenotypeList = cons.env.formatData.phenotypeList #shorter reference to phenotypeList - based on training data (assumes no as yet unseen phenotypes in testing data)
        #----------------------------------------------
        classAccDict = {}
        for each in phenotypeList:
            classAccDict[each] = ClassAccuracy()

        #----------------------------------------------
        if isTrain:
            instances = cons.env.formatData.numTrainInstances
        else:
            instances = cons.env.formatData.numTestInstances
        #----------------------------------------------------------------------------------------------
        for inst in range(instances):
            if isTrain:
                state_phenotype = cons.env.getTrainInstance()
            else:
                state_phenotype = cons.env.getTestInstance()
            #-----------------------------------------------------------------------------
            self.population.makeEvalMatchSet(state_phenotype[0])
            prediction = Prediction(self.population)
            phenotypeSelection = prediction.getDecision() 
            #-----------------------------------------------------------------------------
            
            if phenotypeSelection == None: 
                noMatch += 1
            elif phenotypeSelection == 'Tie':
                tie += 1
            else: #Instances which failed to be covered are excluded from the initial accuracy calculation (this is important to the rule compaction algorithm)
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
        predictionMade = 1.0 - (predictionFail + predictionTies)
        
        adjustedBalancedAccuracy = (balancedAccuracy * predictionMade) + ((1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))
        cons.env.stopEvaluationMode()
        return adjustedBalancedAccuracy
    
    
    def accuracySort(self, cl):
        return cl.accuracy


    def numerositySort(self, cl):
        """ Sorts from smallest numerosity to largest """
        return cl.numerosity
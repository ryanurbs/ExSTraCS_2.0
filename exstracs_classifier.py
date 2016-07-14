"""
Name:        ExSTraCS_Classifier.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: This module defines an individual classifier within the rule population, along with all respective parameters.
             Also included are classifier-level methods, including constructors(covering, copy, reboot) matching, subsumption, 
             crossover, and mutation.  Parameter update methods are also included.
             
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

from exstracs_constants import *
import random
import copy
import math
import ast

class Classifier:
    def __init__(self,a=None,b=None,c=None,d=None):
        #Major Parameters --------------------------------------------------
        self.specifiedAttList = []      # Attribute Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.condition = []             # States of Attributes Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.phenotype = None           # Class if the endpoint is discrete, and a continuous phenotype if the endpoint is continuous
        
        self.fitness = cons.init_fit    # Classifier fitness - initialized to a constant initial fitness value
        self.accuracy = 0.0             # Classifier accuracy - Accuracy calculated using only instances in the dataset which this rule matched.
        self.numerosity = 1             # The number of rule copies stored in the population.  (Indirectly stored as incremented numerosity)
        self.aveMatchSetSize = None     # A parameter used in deletion which reflects the size of match sets within this rule has been included.
        self.deletionVote = None        # The current deletion weight for this classifier.

        #Experience Management ---------------------------------------------
        self.timeStampGA = None         # Time since rule last in a correct set.
        self.initTimeStamp = None       # Iteration in which the rule first appeared.
        self.epochComplete = False      # Has this rule existed for a complete epoch (i.e. a cycle through training set).
        
        #Classifier Accuracy Tracking --------------------------------------
        self.matchCount = 0             # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.correctCount = 0           # The total number of times this classifier was in a correct set
        self.matchCover = 0             # The total number of times this classifier was in a match set within a single epoch. (value fixed after epochComplete)
        self.correctCover = 0           # The total number of times this classifier was in a correct set within a single epoch. (value fixed after epochComplete)
        
        if isinstance(c,list):
            self.classifierCovering(a,b,c,d)
        elif isinstance(a,Classifier):
            self.classifierCopy(a, b)
        elif isinstance(a,list) and b == None:
            self.rebootClassifier(a)
        else:
            print("Classifier: Error building classifier.")
            
            
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER CONSTRUCTION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     
    def classifierCovering(self, setSize, exploreIter, state, phenotype):
        """ Makes a new classifier when the covering mechanism is triggered.  The new classifier will match the current training instance. 
        Covering will NOT produce a default rule (i.e. a rule with a completely general condition). """
        #Initialize new classifier parameters----------
        self.timeStampGA = exploreIter
        self.initTimeStamp = exploreIter
        self.aveMatchSetSize = setSize
        dataInfo = cons.env.formatData
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if dataInfo.discretePhenotype: 
            self.phenotype = phenotype
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        else: 
            print("Classifier - Error: ExSTraCS 2.0 can not handle continuous endpoints.")     

        #-------------------------------------------------------
        # GENERATE MATCHING CONDITION - With Expert Knowledge Weights
        #-------------------------------------------------------     
        #DETERMINISTIC STRATEGY
        if cons.useExpertKnowledge:  
            toSpecify = random.randint(1,dataInfo.specLimit) # Pick number of attributes to specify
            i = 0

            while len(self.specifiedAttList) < toSpecify:
                target = cons.EK.EKRank[i]
                if state[target] != cons.labelMissingData: # If one of the randomly selected specified attributes turns out to be a missing data point, generalize instead.
                    self.specifiedAttList.append(target)
                    self.condition.append(self.buildMatch(target, state)) 
                i += 1
                
        #-------------------------------------------------------
        # GENERATE MATCHING CONDITION - Without Expert Knowledge Weights
        #-------------------------------------------------------
        else: 
            toSpecify = random.randint(1,dataInfo.specLimit) # Pick number of attributes to specify
            potentialSpec = random.sample(range(dataInfo.numAttributes),toSpecify) # List of possible specified attributes
            for attRef in potentialSpec:
                if state[attRef] != cons.labelMissingData: # If one of the randomly selected specified attributes turns out to be a missing data point, generalize instead.
                    self.specifiedAttList.append(attRef)
                    self.condition.append(self.buildMatch(attRef, state))

    def selectAttributeRW(self, toSpecify):
        """ Selects attributes to be specified in classifier covering using Expert Knowledge weights, and roulette wheel selection. """
        scoreRefList = copy.deepcopy(cons.EK.refList) #correct set is a list of reference IDs
        selectList = []
        currentCount = 0  
        totalSum = copy.deepcopy(cons.EK.EKSum)
        while currentCount < toSpecify:
            choicePoint = random.random() * totalSum
            i=0
            sumScore = cons.EK.scores[scoreRefList[i]]
            while choicePoint > sumScore:
                i=i+1
                sumScore += cons.EK.scores[scoreRefList[i]]
            selectList.append(scoreRefList[i]) 
            totalSum -= cons.EK.scores[scoreRefList[i]]
            scoreRefList.remove(scoreRefList[i])
            currentCount += 1
        return selectList 
    
    
    def classifierCopy(self, clOld, exploreIter):
        """  Constructs an identical Classifier.  However, the experience of the copy is set to 0 and the numerosity 
        is set to 1 since this is indeed a new individual in a population. Used by the genetic algorithm to generate 
        offspring based on parent classifiers."""
        self.specifiedAttList = copy.deepcopy(clOld.specifiedAttList)
        self.condition = copy.deepcopy(clOld.condition) 
        self.phenotype = copy.deepcopy(clOld.phenotype)
        self.timeStampGA = exploreIter
        self.initTimeStamp = exploreIter
        self.aveMatchSetSize = copy.deepcopy(clOld.aveMatchSetSize)
        self.fitness = clOld.fitness
        self.accuracy = clOld.accuracy
        
        
    def rebootClassifier(self, classifierList): 
        """ Rebuilds a saved classifier as part of the population Reboot """
        self.specifiedAttList = ast.literal_eval(classifierList[0])
        self.condition = ast.literal_eval(classifierList[1])
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------        
        if cons.env.formatData.discretePhenotype:
            self.phenotype = str(classifierList[2])
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        else:
            print("Classifier - Error: ExSTraCS 2.0 can not handle continuous endpoints.")

        self.fitness = float(classifierList[3])
        self.accuracy = float(classifierList[4])
        self.numerosity = int(classifierList[5])
        self.aveMatchSetSize = float(classifierList[6])
        self.timeStampGA = int(classifierList[7])
        self.initTimeStamp = int(classifierList[8])
        
        if str(classifierList[10]) == 'None':
            self.deletionVote = None
        else:
            self.deletionVote = float(classifierList[10])
        self.correctCount = int(classifierList[11])
        self.matchCount = int(classifierList[12])
        self.correctCover = int(classifierList[13])
        self.matchCover = int(classifierList[14])
        self.epochComplete = bool(classifierList[15])

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # MATCHING
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def match(self, state):
        """ Returns if the classifier matches in the current situation. """ 
        for i in range(len(self.condition)):
            attributeInfo = cons.env.formatData.attributeInfo[self.specifiedAttList[i]]
            #-------------------------------------------------------
            # CONTINUOUS ATTRIBUTE
            #-------------------------------------------------------
            if attributeInfo[0]: 
                instanceValue = state[self.specifiedAttList[i]]
                if self.condition[i][0] < instanceValue < self.condition[i][1] or instanceValue == cons.labelMissingData:
                    pass
                else:
                    return False  
            #-------------------------------------------------------
            # DISCRETE ATTRIBUTE
            #-------------------------------------------------------
            else: 
                stateRep = state[self.specifiedAttList[i]]  
                if stateRep == self.condition[i] or stateRep == cons.labelMissingData:
                    pass
                else:
                    return False 
        return True
        
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GENETIC ALGORITHM MECHANISMS
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def uniformCrossover(self, cl):
        """ Applies uniform crossover and returns if the classifiers changed. Handles both discrete and continuous attributes.  
        #SWARTZ: self. is where for the better attributes are more likely to be specified
        #DEVITO: cl. is where less useful attribute are more likely to be specified
        """
        if cons.env.formatData.discretePhenotype or random.random() < 0.5: #Always crossover condition if the phenotype is discrete (if continuous phenotype, half the time phenotype crossover is performed instead)
            p_self_specifiedAttList = copy.deepcopy(self.specifiedAttList)
            p_cl_specifiedAttList = copy.deepcopy(cl.specifiedAttList)

            useAT = False
            if cons.doAttributeFeedback and random.random() < cons.AT.percent:
                useAT = True
                
            #Make list of attribute references appearing in at least one of the parents.-----------------------------
            comboAttList = []
            for i in p_self_specifiedAttList:
                comboAttList.append(i)
            for i in p_cl_specifiedAttList:
                if i not in comboAttList:
                    comboAttList.append(i)
                elif not cons.env.formatData.attributeInfo[i][0]: #Attribute specified in both parents, and the attribute is discrete (then no reason to cross over)
                    comboAttList.remove(i)
            comboAttList.sort()
            #--------------------------------------------------------------------------------------------------------
            changed = False;   
            for attRef in comboAttList:  
                attributeInfo = cons.env.formatData.attributeInfo[attRef]
                #-------------------------------------------------------
                # ATTRIBUTE CROSSOVER PROBAILITY - ATTRIBUTE FEEDBACK
                #-------------------------------------------------------
                if useAT:
                    probability = cons.AT.getTrackProb()[attRef]
                #-------------------------------------------------------
                # ATTRIBUTE CROSSOVER PROBAILITY - NORMAL CROSSOVER
                #-------------------------------------------------------
                else:
                    probability = 0.5  #Equal probability for attribute alleles to be exchanged.
                #-----------------------------
                ref = 0
                if attRef in p_self_specifiedAttList:
                    ref += 1
                if attRef in p_cl_specifiedAttList:
                    ref += 1

                if ref == 0:    #This should never happen:  All attributes in comboAttList should be specified in at least one classifier.
                    print("Error: UniformCrossover!")
                    pass
                #-------------------------------------------------------
                # CROSSOVER
                #-------------------------------------------------------
                elif ref == 1:  #Attribute specified in only one condition - do probabilistic switch of whole attribute state (Attribute type makes no difference)
                    if attRef in p_self_specifiedAttList and random.random() > probability: # If attribute specified in SWARTZ and high probability of being valuable, then less likely to swap.
                        i = self.specifiedAttList.index(attRef) #reference to the position of the attribute in the rule representation
                        cl.condition.append(self.condition.pop(i)) #Take attribute from self and add to cl
                        cl.specifiedAttList.append(attRef)
                        self.specifiedAttList.remove(attRef)
                        changed = True #Remove att from self and add to cl

                    if attRef in p_cl_specifiedAttList and random.random() < probability: # If attribute specified in DEVITO and high probability of being valuable, then more likely to swap.
                        i = cl.specifiedAttList.index(attRef) #reference to the position of the attribute in the rule representation
                        self.condition.append(cl.condition.pop(i)) #Take attribute from self and add to cl
                        self.specifiedAttList.append(attRef)
                        cl.specifiedAttList.remove(attRef)
                        changed = True #Remove att from cl and add to self.
    
                else: #Attribute specified in both conditions - do random crossover between state alleles - Notice: Attribute Feedback must not be used to push alleles together within an attribute state.
                    #The same attribute may be specified at different positions within either classifier
                    #-------------------------------------------------------
                    # CONTINUOUS ATTRIBUTE
                    #-------------------------------------------------------
                    if attributeInfo[0]: 
                        i_cl1 = self.specifiedAttList.index(attRef) #pairs with self (classifier 1)
                        i_cl2 = cl.specifiedAttList.index(attRef)   #pairs with cl (classifier 2)
                        tempKey = random.randint(0,3) #Make random choice between 4 scenarios, Swap minimums, Swap maximums, Self absorbs cl, or cl absorbs self.
                        if tempKey == 0:    #Swap minimum
                            temp = self.condition[i_cl1][0]
                            self.condition[i_cl1][0] = cl.condition[i_cl2][0]
                            cl.condition[i_cl2][0] = temp
                        elif tempKey == 1:  #Swap maximum
                            temp = self.condition[i_cl1][1]
                            self.condition[i_cl1][1] = cl.condition[i_cl2][1]
                            cl.condition[i_cl2][1] = temp
                        else: #absorb range
                            allList = self.condition[i_cl1] + cl.condition[i_cl2]
                            newMin = min(allList)
                            newMax = max(allList)
                            if tempKey == 2:  #self absorbs cl
                                self.condition[i_cl1] = [newMin,newMax]
                                #Remove cl
                                cl.condition.pop(i_cl2)
                                cl.specifiedAttList.remove(attRef)
                            else:             #cl absorbs self
                                cl.condition[i_cl2] = [newMin,newMax]
                                #Remove self
                                self.condition.pop(i_cl1)
                                self.specifiedAttList.remove(attRef)
                    #-------------------------------------------------------
                    # DISCRETE ATTRIBUTE
                    #-------------------------------------------------------
                    else:
                        pass
                    
            #-------------------------------------------------------
            # SPECIFICATION LIMIT CHECK - return specificity to limit. Note that it is possible for completely general rules to result from crossover - (mutation will ensure that some attribute becomes specified.)
            #-------------------------------------------------------      
            if len(self.specifiedAttList) > cons.env.formatData.specLimit:
                self.specLimitFix(self)

            if len(cl.specifiedAttList) > cons.env.formatData.specLimit:
                self.specLimitFix(cl)
                        
            tempList1 = copy.deepcopy(p_self_specifiedAttList)
            tempList2 = copy.deepcopy(cl.specifiedAttList)
            tempList1.sort()
            tempList2.sort()
            if changed and (tempList1 == tempList2):
                changed = False
            return changed
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE CROSSOVER
        #-------------------------------------------------------
        else: 
            print("Classifier - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
        
      
    def specLimitFix(self,cl):
        """ Lowers classifier specificity to specificity limit. """
        if cons.doAttributeFeedback:
            # Identify 'toRemove' attributes with lowest AT scores
            while len(cl.specifiedAttList) > cons.env.formatData.specLimit:
                minVal = cons.AT.getTrackProb()[cl.specifiedAttList[0]]
                minAtt = cl.specifiedAttList[0]
                for j in cl.specifiedAttList:
                    if cons.AT.getTrackProb()[j] < minVal:
                        minVal = cons.AT.getTrackProb()[j]
                        minAtt = j
                i = cl.specifiedAttList.index(minAtt) #reference to the position of the attribute in the rule representation
                cl.specifiedAttList.remove(minAtt)
                cl.condition.pop(i) #buildMatch handles both discrete and continuous attributes

        else:
            #Randomly pick 'toRemove'attributes to be generalized
            toRemove = len(cl.specifiedAttList) - cons.env.formatData.specLimit
            genTarget = random.sample(cl.specifiedAttList,toRemove)
            for j in genTarget:
                i = cl.specifiedAttList.index(j) #reference to the position of the attribute in the rule representation
                cl.specifiedAttList.remove(j)
                cl.condition.pop(i) #buildMatch handles both discrete and continuous attributes
                        
        
    def Mutation(self, state, phenotype):
        """ Mutates the condition of the classifier. Also handles phenotype mutation. This is a niche mutation, which means that the resulting classifier will still match the current instance.  """
        pressureProb = 0.5 #Probability that if EK is activated, it will be applied.
        useAT = False
        if cons.doAttributeFeedback and random.random() < cons.AT.percent:
            useAT = True
        changed = False;   
        #-------------------------------------------------------
        # MUTATE CONDITION - mutation rate (upsilon) used to probabilistically determine the number of attributes that will be mutated in the classifier.
        #-------------------------------------------------------
        
        steps = 0
        keepGoing = True
        while keepGoing:
            if random.random() < cons.upsilon:
                steps += 1
            else:
                keepGoing = False      

        #Define Spec Limits
        if (len(self.specifiedAttList) - steps) <= 1:
            lowLim = 1
        else:
            lowLim = len(self.specifiedAttList) - steps
        if (len(self.specifiedAttList) + steps) >= cons.env.formatData.specLimit:
            highLim = cons.env.formatData.specLimit
        else:
            highLim = len(self.specifiedAttList) + steps
        if len(self.specifiedAttList) == 0:
            highLim = 1

        #Get new rule specificity.
        newRuleSpec = random.randint(lowLim,highLim)
        #-------------------------------------------------------
        # MAINTAIN SPECIFICITY - 
        #-------------------------------------------------------
        if newRuleSpec == len(self.specifiedAttList) and random.random() < (1-cons.upsilon): #Pick one attribute to generalize and another to specify.  Keep overall rule specificity the same.
            #Identify Generalizing Target
            if not cons.useExpertKnowledge or random.random() > pressureProb:
                genTarget = random.sample(self.specifiedAttList,1)
            else:
                genTarget = self.selectGeneralizeRW(1)
                
            attributeInfo = cons.env.formatData.attributeInfo[genTarget[0]]
            if not attributeInfo[0] or random.random() > 0.5: #GEN/SPEC OPTION
                if not useAT or random.random() > cons.AT.getTrackProb()[genTarget[0]]:
                    #Generalize Target
                    i = self.specifiedAttList.index(genTarget[0]) #reference to the position of the attribute in the rule representation
                    self.specifiedAttList.remove(genTarget[0])
                    self.condition.pop(i) #buildMatch handles both discrete and continuous attributes  
                    changed = True
            else:
                self.mutateContinuousAttributes(useAT,genTarget[0])
            
            #Identify Specifying Target  
            if len(self.specifiedAttList) >= len(state): #Catch for small datasets - if all attributes already specified at this point.
                pass
            else:
                if not cons.useExpertKnowledge or random.random() > pressureProb:
                    pickList = list(range(cons.env.formatData.numAttributes))
                    for i in self.specifiedAttList: # Make list with all non-specified attributes
                        pickList.remove(i)
                    
                    specTarget = random.sample(pickList,1)
                else:
                    specTarget = self.selectSpecifyRW(1)
                if state[specTarget[0]] != cons.labelMissingData and (not useAT or random.random() < cons.AT.getTrackProb()[specTarget[0]]):
                    #Specify Target
                    self.specifiedAttList.append(specTarget[0])
                    self.condition.append(self.buildMatch(specTarget[0], state)) #buildMatch handles both discrete and continuous attributes
                    changed = True
                    
                if len(self.specifiedAttList) > cons.env.formatData.specLimit:    #Double Check
                    self.specLimitFix(self)
        #-------------------------------------------------------
        # INCREASE SPECIFICITY
        #-------------------------------------------------------
        elif newRuleSpec > len(self.specifiedAttList): #Specify more attributes
            change = newRuleSpec - len(self.specifiedAttList)
            if not cons.useExpertKnowledge or random.random() > pressureProb:
                pickList = list(range(cons.env.formatData.numAttributes))
                for i in self.specifiedAttList: # Make list with all non-specified attributes
                    pickList.remove(i)
                specTarget = random.sample(pickList,change)
            else:
                specTarget = self.selectSpecifyRW(change)
            for j in specTarget:
                if state[j] != cons.labelMissingData and (not useAT or random.random() < cons.AT.getTrackProb()[j]):
                    #Specify Target
                    self.specifiedAttList.append(j)
                    self.condition.append(self.buildMatch(j, state)) #buildMatch handles both discrete and continuous attributes
                    changed = True
            
        #-------------------------------------------------------
        # DECREASE SPECIFICITY
        #-------------------------------------------------------
        elif newRuleSpec < len(self.specifiedAttList): # Generalize more attributes.
            change = len(self.specifiedAttList) - newRuleSpec
            if not cons.useExpertKnowledge or random.random() > pressureProb:
                genTarget = random.sample(self.specifiedAttList,change)
            else:
                genTarget = self.selectGeneralizeRW(change)

            #-------------------------------------------------------
            # DISCRETE OR CONTINUOUS ATTRIBUTE - remove attribute specification with 50% chance if we have continuous attribute, or 100% if discrete attribute.
            #-------------------------------------------------------
            for j in genTarget:
                attributeInfo = cons.env.formatData.attributeInfo[j]
                if not attributeInfo[0] or random.random() > 0.5: #GEN/SPEC OPTION
                    if not useAT or random.random() > cons.AT.getTrackProb()[j]:
                        i = self.specifiedAttList.index(j) #reference to the position of the attribute in the rule representation
                        self.specifiedAttList.remove(j)
                        self.condition.pop(i) #buildMatch handles both discrete and continuous attributes
                        changed = True
                else:
                    self.mutateContinuousAttributes(useAT,j)
        else:#Neither specify or generalize.
            pass
        
        #-------------------------------------------------------
        # MUTATE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.formatData.discretePhenotype:
            pass
        else:
            print("Classifier - Error: ExSTraCS 2.0 can not handle continuous endpoints.")

        if changed:# or nowChanged:
            return True


    def selectGeneralizeRW(self, count):
        """ EK applied to the selection of an attribute to generalize for mutation. """
        EKScoreSum = 0
        selectList = []
        currentCount = 0
        specAttList = copy.deepcopy(self.specifiedAttList)
        for i in self.specifiedAttList:
            #When generalizing, EK is inversely proportional to selection probability
            EKScoreSum += 1 / float(cons.EK.scores[i]+1)
            
        while currentCount < count:
            choicePoint = random.random() * EKScoreSum
            i=0
            sumScore = 1 / float(cons.EK.scores[specAttList[i]]+1)
            while choicePoint > sumScore:
                i=i+1
                sumScore += 1 / float(cons.EK.scores[specAttList[i]]+1)
            selectList.append(specAttList[i]) 
            EKScoreSum -= 1 / float(cons.EK.scores[specAttList[i]]+1)
            specAttList.pop(i)
            currentCount += 1
        return selectList  
    
    
    def selectSpecifyRW(self, count):
        """ EK applied to the selection of an attribute to specify for mutation. """
        pickList = list(range(cons.env.formatData.numAttributes))
        for i in self.specifiedAttList: # Make list with all non-specified attributes
            pickList.remove(i)
            
        EKScoreSum = 0
        selectList = []
        currentCount = 0

        for i in pickList:
            #When generalizing, EK is inversely proportional to selection probability
            EKScoreSum += cons.EK.scores[i]
            
        while currentCount < count:
            choicePoint = random.random() * EKScoreSum
            i=0
            sumScore = cons.EK.scores[pickList[i]]
            while choicePoint > sumScore:
                i=i+1
                sumScore += cons.EK.scores[pickList[i]]
            selectList.append(pickList[i]) 
            EKScoreSum -= cons.EK.scores[pickList[i]]
            pickList.pop(i)
            currentCount += 1
        return selectList  
    

    def mutateContinuousAttributes(self, useAT, j):
        #-------------------------------------------------------
        # MUTATE CONTINUOUS ATTRIBUTES
        #-------------------------------------------------------
        if useAT:
            if random.random() < cons.AT.getTrackProb()[j]: #High AT probability leads to higher chance of mutation (Dives ExSTraCS to explore new continuous ranges for important attributes)
                #Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
                attRange = float(cons.env.formatData.attributeInfo[j][1][1]) - float(cons.env.formatData.attributeInfo[j][1][0])
                i = self.specifiedAttList.index(j) #reference to the position of the attribute in the rule representation
                mutateRange = random.random()*0.5*attRange
                if random.random() > 0.5: #Mutate minimum 
                    if random.random() > 0.5: #Add
                        self.condition[i][0] += mutateRange
                    else: #Subtract
                        self.condition[i][0] -= mutateRange
                else: #Mutate maximum
                    if random.random() > 0.5: #Add
                        self.condition[i][1] += mutateRange
                    else: #Subtract
                        self.condition[i][1] -= mutateRange
                #Repair range - such that min specified first, and max second.
                self.condition[i].sort()
                changed = True
        elif random.random() > 0.5:
                #Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
                attRange = float(cons.env.formatData.attributeInfo[j][1][1]) - float(cons.env.formatData.attributeInfo[j][1][0])
                i = self.specifiedAttList.index(j) #reference to the position of the attribute in the rule representation
                mutateRange = random.random()*0.5*attRange
                if random.random() > 0.5: #Mutate minimum 
                    if random.random() > 0.5: #Add
                        self.condition[i][0] += mutateRange
                    else: #Subtract
                        self.condition[i][0] -= mutateRange
                else: #Mutate maximum
                    if random.random() > 0.5: #Add
                        self.condition[i][1] += mutateRange
                    else: #Subtract
                        self.condition[i][1] -= mutateRange
                #Repair range - such that min specified first, and max second.
                self.condition[i].sort()
                changed = True
        else:
            pass
                

    
    
    def rangeCheck(self):
        """ Checks and prevents the scenario where a continuous attributes specified in a rule has a range that fully encloses the training set range for that attribute."""
        for attRef in self.specifiedAttList:
            if cons.env.formatData.attributeInfo[attRef][0]: #Attribute is Continuous
                trueMin = cons.env.formatData.attributeInfo[attRef][1][0]
                trueMax = cons.env.formatData.attributeInfo[attRef][1][1]
                i = self.specifiedAttList.index(attRef)
                valBuffer = (trueMax-trueMin)*0.1 
                if self.condition[i][0] <= trueMin and self.condition[i][1] >= trueMax: # Rule range encloses entire training range
                    self.specifiedAttList.remove(attRef)
                    self.condition.pop(i) 
                    return
                elif self.condition[i][0]+valBuffer < trueMin:
                    self.condition[i][0] = trueMin - valBuffer
                elif self.condition[i][1]- valBuffer > trueMax:
                    self.condition[i][1] = trueMin + valBuffer
                else:
                    pass  
                        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SUBSUMPTION METHODS
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def subsumes(self, cl):
        """ Returns if the classifier (self) subsumes cl """
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.formatData.discretePhenotype: 
            if cl.phenotype == self.phenotype:
                if self.isSubsumer() and self.isMoreGeneral(cl):
                    return True
            return False
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE -
        #-------------------------------------------------------
        else: 
            print("Classifier - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
        

    def isSubsumer(self):
        """ Returns if the classifier (self) is a possible subsumer. A classifier must have sufficient experience (one epoch) and it must also be as or more accurate than the classifier it is trying to subsume.  """
        if self.matchCount > cons.theta_sub and self.accuracy > cons.acc_sub: #self.getAccuracy() > 0.99:
            return True
        return False
    
    
    def isMoreGeneral(self,cl):
        """ Returns if the classifier (self) is more general than cl. Check that all attributes specified in self are also specified in cl. """ 
        if len(self.specifiedAttList) >= len(cl.specifiedAttList):
            return False
        
        for i in range(len(self.specifiedAttList)): #Check each attribute specified in self.condition
            attributeInfo = cons.env.formatData.attributeInfo[self.specifiedAttList[i]]
            if self.specifiedAttList[i] not in cl.specifiedAttList:
                return False
            #-------------------------------------------------------
            # CONTINUOUS ATTRIBUTE
            #-------------------------------------------------------
            if attributeInfo[0]: 
                otherRef = cl.specifiedAttList.index(self.specifiedAttList[i])
                #If self has a narrower ranger of values than it is a subsumer
                if self.condition[i][0] < cl.condition[otherRef][0]:
                    return False
                if self.condition[i][1] > cl.condition[otherRef][1]:
                    return False
                
        return True
    
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # DELETION METHOD
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    def getDelProp(self, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness/self.numerosity >= cons.delta*meanFitness or self.matchCount < cons.theta_del:
            self.deletionVote = self.aveMatchSetSize*self.numerosity

        elif self.fitness == 0.0:
            self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (cons.init_fit/self.numerosity)
        else:
            self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness/self.numerosity) #note, numerosity seems redundant (look into theory of deletion in LCS.
        return self.deletionVote


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
    def buildMatch(self, attRef, state):
        """ Builds a matching condition element given an attribute to be specified for the classifierCovering method. """
        attributeInfo = cons.env.formatData.attributeInfo[attRef]
        #-------------------------------------------------------
        # CONTINUOUS ATTRIBUTE
        #-------------------------------------------------------
        if attributeInfo[0]:
            attRange = attributeInfo[1][1] - attributeInfo[1][0]
            rangeRadius = random.randint(25,75)*0.01*attRange / 2.0 #Continuous initialization domain radius.
            Low = state[attRef] - rangeRadius
            High = state[attRef] + rangeRadius
            condList = [Low,High] #ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.
        #-------------------------------------------------------
        # DISCRETE ATTRIBUTE
        #-------------------------------------------------------
        else: 
            condList = state[attRef] #State already formatted like GABIL in DataManagement
            
        return condList
     

    def equals(self, cl):  
        """ Returns if the two classifiers are identical in condition and phenotype. This works for discrete or continuous attributes or phenotypes. """ 
        if cl.phenotype == self.phenotype and len(cl.specifiedAttList) == len(self.specifiedAttList): #Is phenotype the same and are the same number of attributes specified - quick equality check first.
            clRefs = sorted(cl.specifiedAttList)
            selfRefs = sorted(self.specifiedAttList)
            if clRefs == selfRefs:
                for i in range(len(cl.specifiedAttList)):
                    tempIndex = self.specifiedAttList.index(cl.specifiedAttList[i])
                    if cl.condition[i] == self.condition[tempIndex]:
                        pass
                    else:
                        return False
                return True
        return False


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PARAMETER UPDATES
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     
    def updateEpochStatus(self, exploreIter):
        """ Determines when a learning epoch has completed (one cycle through training data). """
        if not self.epochComplete and (exploreIter - self.initTimeStamp-1) >= cons.env.formatData.numTrainInstances and cons.offlineData:
            self.epochComplete = True

            
    def updateFitness(self):
        """ Update the fitness parameter. """ 
        if cons.env.formatData.discretePhenotype or (self.phenotype[1]-self.phenotype[0])/cons.env.formatData.phenotypeRange < 0.5:
            self.fitness = pow(self.accuracy, cons.nu)
        else:
            print("Classifier - Error: ExSTraCS 2.0 can not handle continuous endpoints.")

                     
    def updateExperience(self):
        """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
        self.matchCount += 1 
        
        if self.epochComplete: #Once epoch Completed, number of matches for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.matchCover += 1


    def updateCorrect(self):
        """ Increases the correct phenotype tracking by one. Once an epoch has completed, rule accuracy can't change."""
        self.correctCount += 1 
        if self.epochComplete: #Once epoch Completed, number of correct for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.correctCover += 1

        
    def updateNumerosity(self, num):
        """ Alters the numberosity of the classifier.  Notice that num can be negative! """
        self.numerosity += num
        
        
    def updateMatchSetSize(self, matchSetSize): 
        """  Updates the average match set size. """
        if self.matchCount < 1.0 / cons.beta:
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount-1)+ matchSetSize) / float(self.matchCount)
        else:
            self.aveMatchSetSize = self.aveMatchSetSize + cons.beta * (matchSetSize - self.aveMatchSetSize)
    
    
    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts
        
        
    def updateAccuracy(self):
        """ Update the accuracy tracker """
        self.accuracy = self.correctCount / float(self.matchCount)
        
        
    def setAccuracy(self,acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc
        
        
    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit
        
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PRINT CLASSIFIER FOR POPULATION OUTPUT FILE
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def printClassifier(self):
        """ Formats and returns an output string describing this classifier. """ 
        classifierString = ""
        classifierString += str(self.specifiedAttList) + "\t"
        classifierString += str(self.condition) + "\t"
        #-------------------------------------------------------------------------------
        specificity = len(self.condition) / float(cons.env.formatData.numAttributes)
        epoch = 0
        if self.epochComplete:
            epoch = 1
        if cons.env.formatData.discretePhenotype:
            classifierString += str(self.phenotype)+"\t"
        else:
            print("Classifier - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        classifierString += str(self.fitness)+"\t"+str(self.accuracy)+"\t"+str(self.numerosity)+"\t"+str(self.aveMatchSetSize)+"\t"+str(self.timeStampGA)+"\t"+str(self.initTimeStamp)+"\t"+str(specificity)+"\t"
        classifierString += str(self.deletionVote)+"\t"+str(self.correctCount)+"\t"+str(self.matchCount)+"\t"+str(self.correctCover)+"\t"+str(self.matchCover)+"\t"+str(epoch)+"\n"

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return classifierString
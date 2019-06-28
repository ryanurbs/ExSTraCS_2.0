"""
Name:        ExSTraCS_ClassifierSet.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Description: This module handles all classifier sets (population, match set, correct set) along with mechanisms and heuristics that act on these sets.
             
---------------------------------------------------------------------------------------------------------------------------------------------------------
ExSTraCS V1.0: Extended Supervised Tracking and Classifying System - An advanced LCS designed specifically for complex, noisy classification/data mining tasks, 
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
from exstracs_classifier import Classifier
import random
import copy
import sys
#------------------------------------------------------

class ClassifierSet:
    def __init__(self, a=None):
        """ Overloaded initialization: Handles creation of a new population or a rebooted population (i.e. a previously saved population). """
        # Major Parameters-----------------------------------
        self.popSet = []        # List of classifiers/rules
        self.matchSet = []      # List of references to rules in population that match
        self.correctSet = []    # List of references to rules in population that both match and specify correct phenotype
        self.microPopSize = 0   # Tracks the current micro population size, i.e. the population size which takes rule numerosity into account. 
        
        #Evaluation Parameters-------------------------------
        self.aveGenerality = 0.0
        self.expRules = 0.0
        self.attributeSpecList = []
        self.attributeAccList = []
        self.avePhenotypeRange = 0.0
        
        #Set Constructors-------------------------------------
        if a==None:
            self.makePop()  #Initialize a new population
        elif isinstance(a,str):
            self.rebootPop(a) #Initialize a population based on an existing saved rule population
        else:
            print("ClassifierSet: Error building population.")
            
            
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # POPULATION CONSTRUCTOR METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def makePop(self):
        """ Initializes the rule population """
        self.popSet = []
            
            
    def rebootPop(self, remakeFile):
        """ Remakes a previously evolved population from a saved text file. """
        print("Rebooting the following population: " + str(remakeFile)+"_RulePop.txt")
        #*******************Initial file handling**********************************************************
        datasetList = []
        try:       
            f = open(remakeFile+"_RulePop.txt", 'rU')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', remakeFile+"_RulePop.txt")
            raise 
        else:
            self.headerList = f.readline().rstrip('\n').split('\t')   #strip off first row
            for line in f:
                lineList = line.strip('\n').split('\t')
                datasetList.append(lineList)
            f.close()

        #**************************************************************************************************
        for each in datasetList:
            cl = Classifier(each)
            self.popSet.append(cl) #Add classifier to the population
            numerosityRef = 5  #location of numerosity variable in population file.
            self.microPopSize += int(each[numerosityRef])


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER SET CONSTRUCTOR METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def makeMatchSet(self, state_phenotype, exploreIter):
        """ Constructs a match set from the population. Covering is initiated if the match set is empty or a rule with the current correct phenotype is absent. """ 
        #Initial values----------------------------------
        state = state_phenotype[0]
        phenotype = state_phenotype[1]
        doCovering = True # Covering check: Twofold (1)checks that a match is present, and (2) that at least one match dictates the correct phenotype.
        setNumerositySum = 0
        #-------------------------------------------------------
        # MATCHING
        #-------------------------------------------------------
        cons.timer.startTimeMatching()                          
        for i in range(len(self.popSet)):                       # Go through the population
            cl = self.popSet[i]                                 # One classifier at a time
            cl.updateEpochStatus(exploreIter)                   # Note whether this classifier has seen all training data at this point.

            if cl.match(state):                                 # Check for match
                self.matchSet.append(i)                         # If match - add classifier to match set
                setNumerositySum += cl.numerosity               # Increment the set numerosity sum
                #Covering Check--------------------------------------------------------    
                if cons.env.formatData.discretePhenotype:       # Discrete phenotype     
                    if cl.phenotype == phenotype:               # Check for phenotype coverage
                        doCovering = False
                else:                                                                               # Continuous phenotype
                    print("ClassifierSet - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
                        
        cons.timer.stopTimeMatching()               
        #-------------------------------------------------------
        # COVERING
        #-------------------------------------------------------
        while doCovering:
            cons.timer.startTimeCovering()
            newCl = Classifier(setNumerositySum+1,exploreIter, state, phenotype)
            self.addClassifierToPopulation(newCl, True)
            self.matchSet.append(len(self.popSet)-1)  # Add covered classifier to matchset
            doCovering = False
            cons.timer.stopTimeCovering()
        

    def makeCorrectSet(self, phenotype):
        """ Constructs a correct set out of the given match set. """      
        for i in range(len(self.matchSet)):
            ref = self.matchSet[i]
            #-------------------------------------------------------
            # DISCRETE PHENOTYPE
            #-------------------------------------------------------
            if cons.env.formatData.discretePhenotype: 
                if self.popSet[ref].phenotype == phenotype:
                    self.correctSet.append(ref) 
            #-------------------------------------------------------
            # CONTINUOUS PHENOTYPE
            #-------------------------------------------------------
            else:
                print("ClassifierSet - Error: ExSTraCS 2.0 can not handle continuous endpoints.")

                
    def makeEvalMatchSet(self, state):  
        """ Constructs a match set for evaluation purposes which does not activate either covering or deletion. """
        for i in range(len(self.popSet)):       # Go through the population
            cl = self.popSet[i]                 # A single classifier
            if cl.match(state):                 # Check for match
                self.matchSet.append(i)         # Add classifier to match set   
                
                
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER DELETION METHODS
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def deletion(self, exploreIter):
        """ Returns the population size back to the maximum set by the user by deleting rules. """
        cons.timer.startTimeDeletion()               
        while self.microPopSize > cons.N:  
            self.deleteFromPopulation() 
        cons.timer.stopTimeDeletion()  
        

    def deleteFromPopulation(self):
        """ Deletes one classifier in the population.  The classifier that will be deleted is chosen by roulette wheel selection
        considering the deletion vote. Returns the macro-classifier which got decreased by one micro-classifier. """
        meanFitness = self.getPopFitnessSum()/float(self.microPopSize)
 
        #Calculate total wheel size------------------------------
        sumCl = 0.0
        voteList = []
        for cl in self.popSet:
            vote = cl.getDelProp(meanFitness)
            sumCl += vote
            voteList.append(vote)
        #--------------------------------------------------------
        choicePoint = sumCl * random.random() #Determine the choice point

        newSum=0.0
        for i in range(len(voteList)):
            cl = self.popSet[i]
            newSum = newSum + voteList[i]
            if newSum > choicePoint: #Select classifier for deletion
                #Delete classifier----------------------------------
                cl.updateNumerosity(-1)
                self.microPopSize -= 1
                if cl.numerosity < 1: # When all micro-classifiers for a given classifier have been depleted.
                    self.removeMacroClassifier(i)
                    self.deleteFromMatchSet(i) 
                    self.deleteFromCorrectSet(i)
                return

        print("ClassifierSet: No eligible rules found for deletion in deleteFrom population.")
        return


    def removeMacroClassifier(self, ref):
        """ Removes the specified (macro-) classifier from the population. """
        self.popSet.pop(ref)    
        
        
    def deleteFromMatchSet(self, deleteRef):
        """ Delete reference to classifier in population, contained in self.matchSet."""
        if deleteRef in self.matchSet:
            self.matchSet.remove(deleteRef)
        #Update match set reference list--------
        for j in range(len(self.matchSet)):
            ref = self.matchSet[j]
            if ref > deleteRef:
                self.matchSet[j] -= 1

        
    def deleteFromCorrectSet(self, deleteRef):
        """ Delete reference to classifier in population, contained in self.matchSet."""
        if deleteRef in self.correctSet:
            self.correctSet.remove(deleteRef)
        #Update match set reference list--------
        for j in range(len(self.correctSet)):
            ref = self.correctSet[j]
            if ref > deleteRef:
                self.correctSet[j] -= 1
            
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GENETIC ALGORITHM
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def runGA(self, exploreIter, state, phenotype):
        """ The genetic discovery mechanism in ExSTraCS is controlled here. """
        #-------------------------------------------------------
        # GA RUN REQUIREMENT
        #-------------------------------------------------------
        if (exploreIter - self.getIterStampAverage()) < cons.theta_GA:  #Does the correct set meet the requirements for activating the GA?
            return 
        self.setIterStamps(exploreIter) #Updates the iteration time stamp for all rules in the correct set (which the GA operates on).
        changed = False
        #-------------------------------------------------------
        # SELECT PARENTS - Niche GA - selects parents from the correct class
        #-------------------------------------------------------
        cons.timer.startTimeSelection()
        if cons.selectionMethod == "roulette": 
            selectList = self.selectClassifierRW()
            clP1 = selectList[0]
            clP2 = selectList[1]
        elif cons.selectionMethod == "tournament":
            selectList = self.selectClassifierT()
            clP1 = selectList[0]
            clP2 = selectList[1]
        else:
            print("ClassifierSet: Error - requested GA selection method not available.")
        cons.timer.stopTimeSelection()
        #-------------------------------------------------------
        # INITIALIZE OFFSPRING 
        #-------------------------------------------------------
        cl1  = Classifier(clP1, exploreIter)
        if clP2 == None:
            cl2 = Classifier(clP1, exploreIter)
        else:
            cl2 = Classifier(clP2, exploreIter) 
        #-------------------------------------------------------
        # CROSSOVER OPERATOR - Uniform Crossover Implemented (i.e. all attributes have equal probability of crossing over between two parents)
        #-------------------------------------------------------
        if not cl1.equals(cl2) and random.random() < cons.chi:  
            cons.timer.startTimeCrossover()
            changed = cl1.uniformCrossover(cl2)
            cons.timer.stopTimeCrossover()
        #-------------------------------------------------------
        # INITIALIZE KEY OFFSPRING PARAMETERS
        #-------------------------------------------------------
        if changed:
            cl1.setAccuracy((cl1.accuracy + cl2.accuracy)/2.0)
            cl1.setFitness(cons.fitnessReduction * (cl1.fitness + cl2.fitness)/2.0)
            cl2.setAccuracy(cl1.accuracy)
            cl2.setFitness(cl1.fitness)
        else:
            cl1.setFitness(cons.fitnessReduction * cl1.fitness)
            cl2.setFitness(cons.fitnessReduction * cl2.fitness)
        #-------------------------------------------------------
        # MUTATION OPERATOR 
        #-------------------------------------------------------
        cons.timer.startTimeMutation()
        nowchanged = cl1.Mutation(state, phenotype)
        howaboutnow = cl2.Mutation(state, phenotype)
        cons.timer.stopTimeMutation()
        
        #Generalize any continuous attributes that span then entire range observed in the dataset.
        if cons.env.formatData.continuousCount > 0:
            cl1.rangeCheck()
            cl2.rangeCheck()
        #-------------------------------------------------------
        # ADD OFFSPRING TO POPULATION
        #-------------------------------------------------------
        if changed or nowchanged or howaboutnow:
            self.insertDiscoveredClassifiers(cl1, cl2, clP1, clP2, exploreIter) #Includes subsumption if activated.
        

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SELECTION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def selectClassifierRW(self):
        """ Selects parents using roulette wheel selection according to the fitness of the classifiers. """
        setList = copy.deepcopy(self.correctSet) #correct set is a list of reference IDs
        if len(setList) > 2:
            selectList = [None, None]
            currentCount = 0  
            while currentCount < 2:
                fitSum = self.getFitnessSum(setList)
                
                choiceP = random.random() * fitSum
                i=0
                sumCl = self.popSet[setList[i]].fitness
                while choiceP > sumCl:
                    i=i+1
                    sumCl += self.popSet[setList[i]].fitness
                    
                selectList[currentCount] = self.popSet[setList[i]] #store reference to the classifier
                setList.remove(setList[i])
                currentCount += 1
                
        elif len(setList) == 2:
            selectList = [self.popSet[setList[0]],self.popSet[setList[1]]]
        elif len(setList) == 1:
            selectList = [self.popSet[setList[0]],self.popSet[setList[0]]]
        else:
            print("ClassifierSet: Error in parent selection.")
        
        return selectList 
    

    def selectClassifierT(self):
        """  Selects parents using tournament selection according to the fitness of the classifiers. """
        setList = copy.deepcopy(self.correctSet) #correct set is a list of reference IDs
        if len(setList) > 2:
            selectList = [None, None]
            currentCount = 0  
            while currentCount < 2:
                tSize = int(len(setList)*cons.theta_sel)
                posList = random.sample(setList,tSize) 
    
                bestF = 0
                bestC = setList[0]
                for j in posList:
                    if self.popSet[j].fitness > bestF:
                        bestF = self.popSet[j].fitness
                        bestC = j
                setList.remove(bestC) #select without re-sampling
                selectList[currentCount] = self.popSet[bestC]
                currentCount += 1
        elif len(setList) == 2:
            selectList = [self.popSet[setList[0]],self.popSet[setList[1]]]
        elif len(setList) == 1:
            selectList = [self.popSet[setList[0]],self.popSet[setList[0]]]
        else:
            print("ClassifierSet: Error in parent selection.")

        return selectList 
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SUBSUMPTION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def subsumeClassifier(self, cl=None, cl1P=None, cl2P=None):
        """ Tries to subsume a classifier in the parents. If no subsumption is possible it tries to subsume it in the current set. """
        if cl1P!=None and cl1P.subsumes(cl):
            self.microPopSize += 1
            cl1P.updateNumerosity(1)
        elif cl2P!=None and cl2P.subsumes(cl):
            self.microPopSize += 1
            cl2P.updateNumerosity(1)
        else:
            self.subsumeClassifier2(cl); #Try to subsume in the correct set.


    def subsumeClassifier2(self, cl):
        """ Tries to subsume a classifier in the correct set. If no subsumption is possible the classifier is simply added to the population considering
        the possibility that there exists an identical classifier. """
        choices = []
        for ref in self.correctSet:
            if self.popSet[ref].subsumes(cl):
                choices.append(ref)

        if len(choices) > 0: #Randomly pick one classifier to be subsumer
            choice = int(random.random()*len(choices))
            self.popSet[choices[choice]].updateNumerosity(1)
            self.microPopSize += 1
            cons.timer.stopTimeSubsumption()
            return
        
        cons.timer.stopTimeSubsumption()
        self.addClassifierToPopulation(cl,False) #If no subsumer was found, check for identical classifier, if not then add the classifier to the population
        
        
    def doCorrectSetSubsumption(self):
        """ Executes correct set subsumption.  The correct set subsumption looks for the most general subsumer classifier in the correct set
        and subsumes all classifiers that are more specific than the selected one. """
        subsumer = None
        for ref in self.correctSet:
            cl = self.popSet[ref]
            if cl.isSubsumer():
                if subsumer == None or cl.isMoreGeneral(subsumer):
                    subsumer = cl

        if subsumer != None: #If a subsumer was found, subsume all more specific classifiers in the correct set
            i=0
            while i < len(self.correctSet):
                ref = self.correctSet[i]
                if subsumer.isMoreGeneral(self.popSet[ref]):
                    subsumer.updateNumerosity(self.popSet[ref].numerosity)
                    self.removeMacroClassifier(ref)
                    self.deleteFromMatchSet(ref)
                    self.deleteFromCorrectSet(ref)
                    i = i - 1 
                i = i + 1
                
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER KEY METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
    def addClassifierToPopulation(self, cl, covering):
        """ Adds a classifier to the set and increases the numerositySum value accordingly."""
        cons.timer.startTimeAdd()
        oldCl = None
        if not covering:
            oldCl = self.getIdenticalClassifier(cl)
        if oldCl != None: #found identical classifier
            oldCl.updateNumerosity(1)
            self.microPopSize += 1
        else:
            self.popSet.append(cl)
            self.microPopSize += 1
        cons.timer.stopTimeAdd()
        
        
    def insertDiscoveredClassifiers(self, cl1, cl2, clP1, clP2, exploreIter):
        """ Inserts both discovered classifiers keeping the maximal size of the population and possibly doing GA subsumption. 
        Checks for default rule (i.e. rule with completely general condition) prevents such rules from being added to the population. """
        #-------------------------------------------------------
        # SUBSUMPTION
        #-------------------------------------------------------
        if cons.doSubsumption:
            cons.timer.startTimeSubsumption()
            
            if len(cl1.specifiedAttList) > 0: 
                self.subsumeClassifier(cl1, clP1, clP2)
            if len(cl2.specifiedAttList) > 0: 
                self.subsumeClassifier(cl2, clP1, clP2)
        #-------------------------------------------------------
        # ADD OFFSPRING TO POPULATION
        #-------------------------------------------------------
        else: 
            if len(cl1.specifiedAttList) > 0:
                self.addClassifierToPopulation(cl1)
            if len(cl2.specifiedAttList) > 0:
                self.addClassifierToPopulation(cl2)
                

    def updateSets(self, exploreIter):
        """ Updates all relevant parameters in the current match and correct sets. """
        matchSetNumerosity = 0
        for ref in self.matchSet:
            matchSetNumerosity += self.popSet[ref].numerosity
        
        for ref in self.matchSet:
            self.popSet[ref].updateExperience()    
            self.popSet[ref].updateMatchSetSize(matchSetNumerosity) #Moved to match set to be like GHCS
            if ref in self.correctSet:
                self.popSet[ref].updateCorrect()

            self.popSet[ref].updateAccuracy()
            self.popSet[ref].updateFitness()
            
            
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
    def getIterStampAverage(self):
        """ Returns the average of the time stamps in the correct set. """
        sumCl=0.0
        numSum=0.0
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            sumCl += self.popSet[ref].timeStampGA * self.popSet[ref].numerosity
            numSum += self.popSet[ref].numerosity #numerosity sum of correct set
        return sumCl/float(numSum)
    

    def setIterStamps(self, exploreIter):
        """ Sets the time stamp of all classifiers in the set to the current time. The current time
        is the number of exploration steps executed so far.  """
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            self.popSet[ref].updateTimeStamp(exploreIter)


    def getFitnessSum(self, setList):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl=0.0
        for i in range(len(setList)):
            ref = setList[i]
            sumCl += self.popSet[ref].fitness
        return sumCl


    def getPopFitnessSum(self):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl=0.0
        for cl in self.popSet:
            sumCl += cl.fitness *cl.numerosity
        return sumCl
    

    def getIdenticalClassifier(self, newCl):
        """ Looks for an identical classifier in the population. """
        for cl in self.popSet:
            if newCl.equals(cl):
                return cl
        return None
    
    
    def clearSets(self):
        """ Clears out references in the match and correct sets for the next learning iteration. """
        self.matchSet = []
        self.correctSet = []
    
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # EVALUTATION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def runPopAveEval(self, exploreIter):
        """ Determines current generality of population """
        genSum = 0
        agedCount = 0

        for cl in self.popSet:
            genSum += ((cons.env.formatData.numAttributes - len(cl.condition)) / float(cons.env.formatData.numAttributes)) * cl.numerosity
            if (exploreIter - cl.initTimeStamp) > cons.env.formatData.numTrainInstances:
                agedCount += 1
    
        if self.microPopSize == 0:
            self.aveGenerality = 'NA'
            self.expRules = 'NA'
        else:
            self.aveGenerality = genSum / float(self.microPopSize) 
            if cons.offlineData:
                self.expRules = agedCount / float(len(self.popSet))
            else:
                self.expRules = 'NA'
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        if not cons.env.formatData.discretePhenotype: 
            print("ClassifierSet - Error: ExSTraCS 2.0 can not handle continuous endpoints.")
            
        
    def runAttGeneralitySum(self):
        """ Determine the population-wide frequency of attribute specification, and accuracy weighted specification. """
        self.attributeSpecList = []
        self.attributeAccList = []
        for i in range(cons.env.formatData.numAttributes):
            self.attributeSpecList.append(0)
            self.attributeAccList.append(0.0)
        for cl in self.popSet:
            for ref in cl.specifiedAttList: 
                self.attributeSpecList[ref] += cl.numerosity
                self.attributeAccList[ref] += cl.numerosity * cl.accuracy


    def recalculateNumerositySum(self):
        """ Recalculate the NumerositySum after rule compaction. """
        self.microPopSize = 0
        for cl in self.popSet:
            self.microPopSize += cl.numerosity
              

    def getPopTrack(self, accuracy, exploreIter, trackingFrequency):
        """ Returns a formated output string to be printed to the Learn Track output file. """
        trackString = str(exploreIter)+ "\t" + str(len(self.popSet)) + "\t" + str(self.microPopSize) + "\t" + str(accuracy) + "\t" + str(self.aveGenerality) + "\t" + str(self.expRules)  + "\t" + str(cons.timer.returnGlobalTimer())+ "\n"
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.formatData.discretePhenotype: 
            print(("Epoch: "+str(int(exploreIter/trackingFrequency))+"\t Iteration: " + str(exploreIter) + "\t MacroPop: " + str(len(self.popSet))+ "\t MicroPop: " + str(self.microPopSize) + "\t AccEstimate: " + str(accuracy) + "\t AveGen: " + str(self.aveGenerality) + "\t ExpRules: " + str(self.expRules)  + "\t Time: " + str(cons.timer.returnGlobalTimer())))
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        else:
            print("ClassifierSet - Error: ExSTraCS 2.0 can not handle continuous endpoints.")

        return trackString
 
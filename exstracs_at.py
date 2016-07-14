"""
Name:        ExSTraCS_AttributeTracking.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: Handles the storage, update, and application of the attribute tracking and feedback heuristics.  This strategy was proposed and 
             published by Ryan Urbanowicz, Ambrose Granizo-Mackenzie, and Jason Moore in "Instance-Linked Attribute Tracking and Feedback for 
             Michigan-Style Supervised Learning Classifier Systems." [2012].
             
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
import copy
import random
#------------------------------------------------------

class AttributeTracking:
    def __init__(self, doAttTrack):
        """ Initialize Attribute Tracking Object"""
        self.percent = 0.0
        if doAttTrack:
            self.probabilityList = []
            self.attAccuracySums = [[0]*cons.env.formatData.numAttributes for i in range(cons.env.formatData.numTrainInstances)]
            if cons.doPopulationReboot:
                self.rebootAT()

    
    def updatePercent(self, exploreIter):
        """ Determines the frequency with which attribute feedback is applied within the GA.  """
        self.percent = exploreIter/float(cons.maxLearningIterations)
        
           
    def updateAttTrack(self, pop):
        """ Attribute Tracking update."""
        dataRef = cons.env.dataRef
        for ref in pop.correctSet:
            for each in pop.popSet[ref].specifiedAttList:
                self.attAccuracySums[dataRef][each] += (pop.popSet[ref].accuracy) #Add rule accuracy

    
    def getTrackProb(self):
        """ Returns the tracking probability list. """
        return self.probabilityList
       
       
    def genTrackProb(self):
        """ Calculate and return the attribute probabilities based on the attribute tracking scores. """
        #Choose a random data instance attribute tracking scores
        currentInstance = random.randint(0,cons.env.formatData.numTrainInstances-1)
        #Get data set reference.
        trackList = copy.deepcopy(self.attAccuracySums[currentInstance])
        #----------------------------------------
        minVal = min(trackList)
        for i in range(len(trackList)):
            trackList[i] = trackList[i] - minVal
        maxVal = max(trackList)
        #----------------------------------------
        probList = []
        for i in range(cons.env.formatData.numAttributes):
            if maxVal == 0.0:
                probList.append(0.5)
            else:
                probList.append(trackList[i]/float(maxVal + maxVal*0.01))  #perhaps make this float a constant, or think of better way to do this.

        self.probabilityList = probList
      
      
    def sumGlobalAttTrack(self):
        """ For each attribute, sum the attribute tracking scores over all instances. For Reporting and Debugging"""
        globalAttTrack = [0.0 for i in range(cons.env.formatData.numAttributes)]
        for i in range(cons.env.formatData.numAttributes):
            for j in range(cons.env.formatData.numTrainInstances):
                globalAttTrack[i] += self.attAccuracySums[j][i]
        return globalAttTrack


    def rebootAT(self):
        """ Rebuilds attribute tracking scores from previously stored run. """
        try: #Obtain existing attribute tracking
            f = open(cons.popRebootPath+"_AttTrack.txt", 'rU')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', cons.popRebootPath+"_AttTrack.txt")
            raise 
        else:
            junkList = f.readline().rstrip('\n').split('\t')
            ATList = []
            for line in f:
                lineList = line.strip('\n').split('\t')
                ATList.append(lineList)
            f.close()

            #Reorder old att-track values to match new data shuffling.
            dataLink = cons.env.formatData
            for i in range(dataLink.numTrainInstances):
                targetID = dataLink.trainFormatted[i][2] #gets each instance ID
                notFound = True
                j = 0
                while notFound and j < dataLink.numTrainInstances:
                    if str(targetID) == str(ATList[j][0]): #found relevant instance
                        for w in range(dataLink.numAttributes):
                            self.attAccuracySums[i][w] = float(ATList[j][w+1])
                    j += 1              

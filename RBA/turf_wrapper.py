"""
Name:        TurfMe.py
Authors:     Gediminas Bertasius and Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     December 4, 2013
Modified:    August 25,2014
Description: Turf algorithm iterates through running some other relief-based algorithm, each time filtering out a given percentage of the remaining attributes.
This allows for relief-based algorithm scores to be readjusted after filtering out probable noise attributes.
             
---------------------------------------------------------------------------------------------------------------------------------------------------------
ReBATE: includes stand-alone Python code to run any of the included/available Relief-Based algorithms designed for attribute filtering/ranking.
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
#from ExSTraCS_DataManagement import *
from RBA.relieff import *
from RBA.surf import *
from RBA.surfstar import *
from RBA.multisurf import *

class TuRFMe:
    def __init__(self, env, algorithm, turfPercent, reliefSampleFraction, reliefNeighbors):
        self.data = env.formatData
        self.algorithm = algorithm
        self.reliefSampleFraction = reliefSampleFraction
        self.reliefNeighbors = reliefNeighbors
        
        self.filterScores = []
        self.keepManaging = True
        self.keepRunningAlgorithms = True
        self.turfPercent = turfPercent
        self.N = int(1/float(turfPercent)) #Number of iterations
        print("Running turf for "+str(self.N) +" iterations.")

        self.data.saveTempTurfData()
        self.runTurf()
        self.data.returntoFullData()
        env.resetDataRef(True)

    def runTurf(self):
        i = 0
        while i < self.N-1 and self.keepRunningAlgorithms:          
            #Choose and run desired algorithm---------------------------------------------------------
            if self.algorithm=="multisurf_turf":
                self.filterScores = Run_MultiSURF(self.data)
            elif self.algorithm=="surfstar_turf":
                self.filterScores = Run_SURFStar(self.data, self.reliefSampleFraction)
            elif self.algorithm=="surf_turf":
                self.filterScores = Run_SURF(self.data, self.reliefSampleFraction)
            elif self.algorithm=="relieff_turf":
                self.filterScores = Run_ReliefF(self.data, self.reliefSampleFraction, self.reliefNeighbors)
            else:
                print("ERROR: Algorithm not found.")   
            if not self.keepManaging:
                self.keepRunningAlgorithms = False 
            
            if self.keepManaging and not iter == (self.N-1): #Filter the data, all but the last iteration.
                self.keepManaging = self.data.turfDataManagement(self.filterScores, self.turfPercent)
            i+=1
        
        #Find low score
        lowScore = min(self.filterScores)
        maxScore = max(self.filterScores)
        thisrange = maxScore - lowScore
        tierScoreReduction = 0.01*thisrange
        #Define Tier Scores (unique score for all attributes removed at a specific filter tier.
        tierScores = []
        for k in range(len(self.data.tierList)):
            tierScores.append(lowScore-(tierScoreReduction*(k+1)))
   
        tierScores.reverse() #Reversed because the worst tier was first in the tierList
         
        #Cycle through original header list (all attributes, and build a new filterscore list from scratch.
        finalFilterScores = []
        for j in range(len(self.data.turfHeaderList)): #All original Attributes
            #Find where this attribute is (final score list, or one of the removed tiers
            if self.data.turfHeaderList[j] in self.data.trainHeaderList: #Made it to final cut
                scoreID = self.data.trainHeaderList.index(self.data.turfHeaderList[j])
                finalFilterScores.append(self.filterScores[scoreID])
                
            else: #filtered out - must find correct tier.
                for k in range(len(self.data.tierList)):
                    if self.data.turfHeaderList[j] in self.data.tierList[k]:
                        finalFilterScores.append(tierScores[k])
                    
        print(finalFilterScores)
        self.filterScores = finalFilterScores
        
        



#
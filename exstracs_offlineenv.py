"""
Name:        ExSTraCS_Offline_Environement.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: In the context of data mining and classification tasks, the 'environment' for ExSTraCS is a data set with a limited number of instances
             with some number of attributes and a single endpoint (typically a discrete phenotype or class) of interest.  This module manages ExSTraCS's 
             stepping through learning iterations, and data instances respectively.  Special methods are included to jump from learning to evaluation of a 
             training dataset.
             
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
from exstracs_data import DataManagement
from exstracs_constants import * 
import sys
#------------------------------------------------------

class Offline_Environment:
    def __init__(self):
        """  """
        #Initialize global variables-------------------------------------------------
        self.dataRef = 0
        self.storeDataRef = 0
        self.formatData = DataManagement(cons.trainFile, cons.testFile)
        
        self.currentTrainState = self.formatData.trainFormatted[self.dataRef][0]
        self.currentTrainPhenotype = self.formatData.trainFormatted[self.dataRef][1]
        if cons.testFile == 'None':
            pass
        else:
            self.currentTestState = self.formatData.testFormatted[self.dataRef][0]
            self.currentTestPhenotype = self.formatData.testFormatted[self.dataRef][1]
        

    def getTrainInstance(self):
        """ Returns the current training instance. """  #NOTE: Probably faster way of doing this than additional 'if' statement every learning iteration
        return [self.currentTrainState, self.currentTrainPhenotype] #Return unadulterated training data
        
        
    def getTestInstance(self):
        """ Returns the current training instance. """
        return [self.currentTestState, self.currentTestPhenotype]
    
    
    def newInstance(self, isTraining): 
        """  Shifts the environment to the next instance in the data. """
        #-------------------------------------------------------
        # Training Data
        #-------------------------------------------------------
        if isTraining: 
            if self.dataRef < (self.formatData.numTrainInstances-1):
                self.dataRef += 1
                self.currentTrainState = self.formatData.trainFormatted[self.dataRef][0]
                self.currentTrainPhenotype = self.formatData.trainFormatted[self.dataRef][1]

            else: #Once learning has completed an epoch (i.e. learning iterations though the entire dataset) it starts back at the first instance in the data)
                self.resetDataRef(isTraining)
        #-------------------------------------------------------
        # Testing Data
        #-------------------------------------------------------
        else: 
            if self.dataRef < (self.formatData.numTestInstances-1):
                self.dataRef += 1
                self.currentTestState = self.formatData.testFormatted[self.dataRef][0]
                self.currentTestPhenotype = self.formatData.testFormatted[self.dataRef][1]
      
      
    def resetDataRef(self, isTraining):
        """ Resets the iteration count through the current data set. """
        self.dataRef = 0 
        if isTraining:
            self.currentTrainState = self.formatData.trainFormatted[self.dataRef][0]
            self.currentTrainPhenotype = self.formatData.trainFormatted[self.dataRef][1]
        else:
            self.currentTestState = self.formatData.testFormatted[self.dataRef][0]
            self.currentTestPhenotype = self.formatData.testFormatted[self.dataRef][1]


    def startEvaluationMode(self):
        """ Turns on evaluation mode.  Saves the instance we left off in the training data. Also important when using RAIN."""
        self.storeDataRef = self.dataRef
        
        
    def stopEvaluationMode(self):
        """ Turns off evaluation mode.  Re-establishes place in dataset."""
        self.dataRef = self.storeDataRef

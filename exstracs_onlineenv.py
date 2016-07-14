"""
Name:        ExSTraCS_Online_Environement.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: ExSTraCS is best suited to offline iterative learning, however this module has been implemented as an example of how ExSTraCS may be used
             to perform online learning as well.  Here, this module has been written to perform online learning for a n-multiplexer problem, where training
             instances are generated in an online fashion.  This module has not been fully tested.
             
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
from Online_Learning.problem_multiplexer import *  #http://stackoverflow.com/questions/4383571/importing-files-from-different-folder-in-python
import sys
#------------------------------------------------------

class Online_Environment:
    def __init__(self):
        """ Specify source of online data with appropriate method. """
        trainFile = None
        testFile = None
        """ Specifically Designed to get n-bit Mulitplexer problem data
        Valid Multiplexers:
        Address Bits = 1 (3-Multiplexer)
        Address Bits = 2 (6-Multiplexer)
        Address Bits = 3 (11-Multiplexer)   
        Address Bits = 4 (20-Multiplexer)  
        Address Bits = 5 (37-Multiplexer)                      
        Address Bits = 6 (70-Multiplexer)   
        Address Bits = 7 (135-Multiplexer)   
        Address Bits = 8 (264-Multiplexer) 
        """
        #Multiplexer specific variables
        self.num_bits = 6 # E.g. 3, 6, 11, 20...

        infoList = self.mulitplexerInfoList()

        self.formatData = DataManagement(trainFile, testFile, infoList)
        first_Instance = generate_multiplexer_instance(self.num_bits)
        print(first_Instance)
        self.currentTrainState = first_Instance[0]
        self.currentTrainPhenotype = first_Instance[1]

        
    def mulitplexerInfoList(self):
        """ Manually specify all dataset parameters for Multiplexer problem. """      
        numAttributes = self.num_bits
        discretePhenotype = True
        attributeInfo = []
        for i in range(self.num_bits):
            attributeInfo.append([0,[]])

        phenotypeList = ['0','1']
        phenotypeRange = None
        trainHeaderList = []
        for i in range(self.num_bits):
            trainHeaderList.append('X_'+str(i)) #Give online data some arbitrary attribute names.
        numTrainInstances = 0
        infoList = [numAttributes,discretePhenotype,attributeInfo,phenotypeList,phenotypeRange,trainHeaderList,numTrainInstances]
        print(infoList)
        return infoList
        
            
    def newInstance(self,eval): 
        """  Shifts the environment to the next instance in the data. """
        new_Instance = generate_multiplexer_instance(self.num_bits)
        self.currentTrainState = new_Instance[0]
        self.currentTrainPhenotype = new_Instance[1]
         

    def getTrainInstance(self):
        """ Returns the current training instance. """ 
        return [self.currentTrainState, self.currentTrainPhenotype]
        
    def startEvaluationMode(self):
        """ Turns on evaluation mode.  Saves the instance we left off in the training data. Also important when using RAIN."""
        pass
        
        
    def stopEvaluationMode(self):
        """ Turns off evaluation mode.  Re-establishes place in dataset."""
        pass
"""
Name:        ExSTraCS_Timer.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: This module's role is largely for development and evaluation purposes.  Specifically it tracks not only global run time for ExSTraCS, but 
             tracks the time utilized by different key mechanisms of the algorithm.  This tracking likely wastes a bit of run time, so for optimal performance
             check that all 'cons.timer.startXXXX', and 'cons.timer.stopXXXX' commands are commented out within ExSTraCS_Main, ExSTraCS_Test, ExSTraCS_Algorithm, 
             and ExSTraCS_ClassifierSet.
             
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
import time
#------------------------------------------------------

class Timer:
    def __init__(self):
        """ Initializes all Timer values for the algorithm """        
        # Global Time objects
        self.globalStartRef = time.time()
        self.globalTime = 0.0
        self.addedTime = 0.0
        
        # Match Time Variables
        self.startRefMatching = 0.0
        self.globalMatching = 0.0

        # Covering Time Variables
        self.startRefCovering = 0.0
        self.globalCovering = 0.0
        
        # Deletion Time Variables
        self.startRefDeletion = 0.0
        self.globalDeletion = 0.0

        # Subsumption Time Variables
        self.startRefSubsumption = 0.0
        self.globalSubsumption = 0.0

        # Selection Time Variables
        self.startRefSelection = 0.0
        self.globalSelection = 0.0
        
        # Crossover Time Variables
        self.startRefCrossover = 0.0
        self.globalCrossover = 0.0
        
        # Mutation Time Variables
        self.startRefMutation = 0.0
        self.globalMutation = 0.0
        
        # Attribute Tracking and Feedback
        self.startRefAT = 0.0
        self.globalAT = 0.0

        # Expert Knowledge (EK)
        self.startRefEK = 0.0
        self.globalEK = 0.0

        # OutFile
        self.startRefOutFile = 0.0
        self.globalOutFile = 0.0

        # Initialization
        self.startRefInit = 0.0
        self.globalInit = 0.0
        
        # Add Classifier
        self.startRefAdd = 0.0
        self.globalAdd = 0.0
        
        # Evaluation Time Variables
        self.startRefEvaluation = 0.0
        self.globalEvaluation = 0.0  
        
        # Rule Compaction
        self.startRefRuleCmp = 0.0
        self.globalRuleCmp = 0.0

        # Rule Compaction
        self.startRefTEST = 0.0
        self.globalTEST = 0.0
        
        #Debug Counter
        self.globalTESTCounter = 0
        
    # ************************************************************
    def startTimeMatching(self):
        """ Tracks MatchSet Time """
        self.startRefMatching = time.time()
         
    def stopTimeMatching(self):
        """ Tracks MatchSet Time """
        diff = time.time() - self.startRefMatching
        self.globalMatching += diff        

    # ************************************************************
    def startTimeCovering(self):
        """ Tracks MatchSet Time """
        self.startRefCovering = time.time()
         
    def stopTimeCovering(self):
        """ Tracks MatchSet Time """
        diff = time.time() - self.startRefCovering
        self.globalCovering += diff        
        
    # ************************************************************
    def startTimeDeletion(self):
        """ Tracks Deletion Time """
        self.startRefDeletion = time.time()
        
    def stopTimeDeletion(self):
        """ Tracks Deletion Time """
        diff = time.time() - self.startRefDeletion
        self.globalDeletion += diff
    
    # ************************************************************
    def startTimeCrossover(self):
        """ Tracks Crossover Time """
        self.startRefCrossover = time.time() 
               
    def stopTimeCrossover(self):
        """ Tracks Crossover Time """
        diff = time.time() - self.startRefCrossover
        self.globalCrossover += diff
        
    # ************************************************************
    def startTimeMutation(self):
        """ Tracks Mutation Time """
        self.startRefMutation = time.time()
        
    def stopTimeMutation(self):
        """ Tracks Mutation Time """
        diff = time.time() - self.startRefMutation
        self.globalMutation += diff
        
    # ************************************************************
    def startTimeSubsumption(self):
        """Tracks Subsumption Time """
        self.startRefSubsumption = time.time()

    def stopTimeSubsumption(self):
        """Tracks Subsumption Time """
        diff = time.time() - self.startRefSubsumption
        self.globalSubsumption += diff    
        
    # ************************************************************
    def startTimeSelection(self):
        """ Tracks Selection Time """
        self.startRefSelection = time.time()
        
    def stopTimeSelection(self):
        """ Tracks Selection Time """
        diff = time.time() - self.startRefSelection
        self.globalSelection += diff
    
    # ************************************************************
    def startTimeEvaluation(self):
        """ Tracks Evaluation Time """
        self.startRefEvaluation = time.time()
        
    def stopTimeEvaluation(self):
        """ Tracks Evaluation Time """
        diff = time.time() - self.startRefEvaluation
        self.globalEvaluation += diff 
    
    # ************************************************************
    def startTimeRuleCmp(self):
        """  """
        self.startRefRuleCmp = time.time()   
         
    def stopTimeRuleCmp(self):
        """  """
        diff = time.time() - self.startRefRuleCmp
        self.globalRuleCmp += diff
        
    # ***********************************************************  
    def startTimeAT(self):
        """  """
        self.startRefAT = time.time()   
         
    def stopTimeAT(self):
        """  """
        diff = time.time() - self.startRefAT
        self.globalAT += diff
        
    # ***********************************************************
    def startTimeEK(self):
        """  """
        self.startRefEK = time.time()   
         
    def stopTimeEK(self):
        """  """
        diff = time.time() - self.startRefEK
        self.globalEK += diff
        
    # ***********************************************************
    def startTimeOutFile(self):
        """  """
        self.startRefOutFile = time.time()   
         
    def stopTimeOutFile(self):
        """  """
        diff = time.time() - self.startRefOutFile
        self.globalOutFile += diff
        
    # ***********************************************************
    def startTimeInit(self):
        """  """
        self.startRefInit = time.time()   
         
    def stopTimeInit(self):
        """  """
        diff = time.time() - self.startRefInit
        self.globalInit += diff
        
    # ***********************************************************
    def startTimeAdd(self):
        """  """
        self.startRefAdd = time.time()   
         
    def stopTimeAdd(self):
        """  """
        diff = time.time() - self.startRefAdd
        self.globalAdd += diff
        
    # ***********************************************************
    def startTimeTEST(self):
        """  """
        self.startRefTEST = time.time()   
         
    def stopTimeTEST(self):
        """  """
        diff = time.time() - self.startRefTEST
        self.globalTEST += diff
    # ***********************************************************
    
    def returnGlobalTimer(self):
        """ Set the global end timer, call at very end of algorithm. """
        self.globalTime = (time.time() - self.globalStartRef) + self.addedTime #Reports time in minutes, addedTime is for population reboot.
        return self.globalTime/ 60.0
        
        
    def TESTCounter(self):
        """ Set the global end timer, call at very end of algorithm. """
        self.globalTESTCounter += 1

    
    def setTimerRestart(self, remakeFile): 
        """ Sets all time values to the those previously evolved in the loaded popFile.  """
        print(remakeFile+"_PopStats.txt")
        try:
            fileObject = open(remakeFile+"_PopStats.txt", 'rU')  # opens each datafile to read.
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', remakeFile+"_PopStats.txt")
            raise 

        timeDataRef = 22
        tempLine = None
        for i in range(timeDataRef):
            tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')
        self.addedTime = float(tempList[1]) * 60 #previous global time added with Reboot.
        
        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalMatching = float(tempList[1]) * 60

        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalCovering = float(tempList[1]) * 60
        
        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalDeletion = float(tempList[1]) * 60

        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalSubsumption = float(tempList[1]) * 60
        
        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalSelection = float(tempList[1]) * 60    
 
        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalCrossover = float(tempList[1]) * 60

        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalMutation = float(tempList[1]) * 60

        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalAT = float(tempList[1]) * 60
 
        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalEK = float(tempList[1]) * 60

        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalOutFile = float(tempList[1]) * 60

        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalInit = float(tempList[1]) * 60
        
        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalAdd = float(tempList[1]) * 60
        
        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalEvaluation = float(tempList[1]) * 60
                     
        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t') 
        self.globalRuleCmp = float(tempList[1]) * 60
        
        fileObject.close()
        

    def reportTimes(self):
        self.returnGlobalTimer()
        """ Reports the time summaries for this run. Returns a string ready to be printed out."""
        outputTime = "Global Time\t"+str(self.globalTime/ 60.0)+ \
        "\nMatching Time\t" + str(self.globalMatching/ 60.0)+ \
        "\nCovering Time\t" + str(self.globalCovering/ 60.0)+ \
        "\nDeletion Time\t" + str(self.globalDeletion/ 60.0)+ \
        "\nSubsumption Time\t" + str(self.globalSubsumption/ 60.0)+ \
        "\nSelection Time\t"+str(self.globalSelection/ 60.0)+ \
        "\nCrossover Time\t" + str(self.globalCrossover/ 60.0)+ \
        "\nMutation Time\t" + str(self.globalMutation/ 60.0)+ \
        "\nAttribute Tracking-Feedback Time\t"+str(self.globalAT/ 60.0) + \
        "\nExpert Knowledge Time\t"+str(self.globalEK/ 60.0) + \
        "\nOutput File Time\t"+str(self.globalOutFile/ 60.0) + \
        "\nInitialization Time\t"+str(self.globalInit/ 60.0) + \
        "\nAdd Classifier Time\t"+str(self.globalAdd/ 60.0) + \
        "\nEvaluation Time\t"+str(self.globalEvaluation/ 60.0) + \
        "\nRule Compaction Time\t"+str(self.globalRuleCmp/ 60.0) + "\n" 
        
        return outputTime
"""
Name:        ExSTraCS_ClassAccuracy.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: Used for global evaluations of the LCS rule population for problem domains with a discrete phenotype.  Allows for the calculation of balanced
             accuracy when a discrete phenotype includes two or more possible classes.
             
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

class ClassAccuracy:
    def __init__(self):
        """ Initialize the accuracy calculation for a single class """
        self.T_myClass = 0      #true positive for binary class problems
        self.T_otherClass = 0   #true negative for binary class problems
        self.F_myClass = 0      #false positive for binary class problems
        self.F_otherClass = 0   #false negative for binary class problems


    def updateAccuracy(self, thisIsMe, accurateClass):
        """ Increment the appropriate cell of the confusion matrix """
        if thisIsMe and accurateClass:
            self.T_myClass += 1
        elif accurateClass:
            self.T_otherClass += 1
        elif thisIsMe:
            self.F_myClass += 1
        else:
            self.F_otherClass += 1
        
        
    def reportClassAccuracy(self):
        """ Print to standard out, summary on the class accuracy. """
        print("-----------------------------------------------")
        print("TP = "+str(self.T_myClass))
        print("TN = "+str(self.T_otherClass))
        print("FP = "+str(self.F_myClass))
        print("FN = "+str(self.F_otherClass))
        print("-----------------------------------------------")
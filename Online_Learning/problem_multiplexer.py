"""
Name: Problem_Multiplexer.py
Authors: Gediminas Bertasius and Ryan Urbanowicz - written at Dartmouth College, Hanover, NH, USA
Contact: ryan.j.urbanowicz@darmouth.edu
Created: June 13, 2013
---------------------------------------------------------------------------------------------------------------------------------------------------------
Problem_Multiplexer: A script designed to generate toy n-multiplexer problem datasets.  These are a typical scalable toy problem used in classification 
and data mining algorithms such as learning classifier systems.  The 'generate_multiplexer_instance' method will return a single multiplexer instance when called.
The 'generate_multiplexer_data' method will generate a specified number of instances for a given n-multiplexer problem and save them to a file.  
Lastly, 'generate_complete_multiplexer_data' will attempt to generate all possible unique instances of an n-multiplexer problem, assuming there is enough 
memory to complete the task.  This dataset is also saved to a file.  Below we break down the first 8 multiplexer problems, where the number of address bits 
determines the total length of the multiplexer binary string.
Address Bits = 1 (3-Multiplexer)
Address Bits = 2 (6-Multiplexer) - 8 optimal rules - 64 unique instances
Address Bits = 3 (11-Multiplexer) -  16 optimal rules - 2048 unique instances
Address Bits = 4 (20-Multiplexer)  - 32 optimal rules - 1,048,576 unique instances
Address Bits = 5 (37-Multiplexer) - 64 optimal rules - 137,438,953,472L unique instances
Address Bits = 6 (70-Multiplexer) - 128 optimal rules - 1180591620717411303424L unique instances
Address Bits = 7 (135-Multiplexer) - 256 optimal rules - HUGE
Address Bits = 8 (264-Multiplexer) - 512 optimal rules - HUGE

Copyright (C) 2013 Ryan Urbanowicz 
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, 
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import random

def generate_mulitplexer_data(myfile, num_bits, instances):
    """ """
    print("Problem_Multiplexer: Generate multiplexer dataset with "+str(instances)+" instances.")
    first=solve_equation(num_bits)
    if first==None:
        print("Problem_Multiplexer: ERROR - The multiplexer takes # of bits as 3,6,11,20,37,70,135,264")
        
    else:
        fp=open(myfile,"w")
        #Make File Header
        for i in range(first):
            fp.write('A_'+str(i)+"\t") #Address Bits
            
        for i in range(num_bits-first):
            fp.write('R_'+str(i)+"\t") #Register Bits
        fp.write("Class" + "\n") #State found at Register Bit
        
        for i in range(instances):
            state_phenotype = generate_multiplexer_instance(num_bits)
            for j in state_phenotype[0]:
                fp.write(str(j)+"\t")
            fp.write(str(state_phenotype[1])+ "\n")
        

def generate_multiplexer_instance(num_bits):
    """ """
    first=solve_equation(num_bits)
    if first==None:
        print("Problem_Multiplexer: ERROR - The multiplexer takes # of bits as 3,6,11,20,37,70,135,264")
        
    else:
        condition = []
        #Generate random boolean string
        for i in range(num_bits):
            condition.append(str(random.randint(0,1)))
            
        gates=""
        
        for j in range(first):
            gates+=condition[j]
        
        gates_decimal=int(gates,2)
        output=condition[first+gates_decimal]

        return [condition,output]



def generate_complete_multiplexer_data(myfile,num_bits):
    """ Attempts to generate a complete non-redundant multiplexer dataset.  Ability to generate the entire dataset is computationally limited. 
     We had success generating up to the complete 20-multiplexer dataset"""
     
    print("Problem_Multiplexer: Attempting to generate multiplexer dataset")
    first=solve_equation(num_bits)
    
    if first==None:
        print("Problem_Multiplexer: ERROR - The multiplexer takes # of bits as 3,6,11,20,37,70,135,264")
    else:
        try:
            fp=open(myfile,"w")
            for i in range(2**num_bits):
                binary_str=bin(i)
                string_array=binary_str.split('b')
                binary=string_array[1]
                
                while len(binary)<num_bits:
                    binary="0" + binary
                    
                gates=""
                for j in range(first):
                    gates+=binary[j]
                
                gates_decimal=int(gates,2)
                output=binary[first+gates_decimal]
                
                fp.write(str(i)+"\t")
                fp.write(binary+ "\t")
                fp.write(output+ "\n")
                
            fp.close()
            print("Problem_Multiplexer: Dataset Generation Complete")
            
        except:
            print("Problem_Multiplexer: ERROR - Cannot generate all data instances for specified multiplexer due to computational limitations")
            
            
def solve_equation(num_bits):
    for i in range(1000):
        if i+2**i==num_bits:
            return i
    return None

#generate_multiplexer_instance(37)
#generate_mulitplexer_data("Multiplexer_Data.txt", 135, 40000)
#generate_complete_multiplexer_data("Multiplexer_Data.txt",37)  #3,6,11,20, 37

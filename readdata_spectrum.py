import os
import numpy as np
from tqdm import tqdm
import sys

def load_data(filename,labelfile,kinds,fault_type):
    data = np.zeros((kinds,100,100,3))
    label = np.zeros((kinds, fault_type))
    name = list()
    
    count = 0
    with open(filename) as f:
        for line in tqdm(f):
            temp = line.strip().split()
            temp = np.array(temp)
            kind = count//3
            phase = count-kind*3
            data[kind,:,:,phase] = temp.reshape(100,100)
            count = count + 1

    count = 0
    with open(labelfile) as f2:
        for line in tqdm(f2):
            temp = line.strip().split()
            label[count, 0] = int(temp[3])
            label[count, 1] = int(temp[4])
            label[count, 2] = int(temp[5])
            label[count, 3] = int(temp[6])
            label[count, 4] = int(temp[7])
            label[count, 5] = int(temp[8])
            label[count, 6] = int(temp[9])
            label[count, 7] = int(temp[10])
            label[count, 8] = int(temp[11])

            if (fault_type==11):
                label[count, 9] = int(temp[12]) 
                if temp[13] == "1.0":           #location
                    label[count, 10] = 0.0
                else:
                    label[count, 10] = float(temp[13])
                name.append(temp[14])
            
            elif (fault_type ==10):
                label[count, 9] = float(temp[12]) 
                name.append(temp[13])

            count = count + 1
    print (str(count) + 'lines are read')
    return [data, label,name]

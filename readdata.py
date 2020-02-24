import os
import numpy as np
from tqdm import tqdm

def load_data(filename,length,kinds,fault_type):
    data = np.zeros((length * kinds, 3))
    label = np.zeros((length * kinds, fault_type))
    name = list()
    
    count = 0
    with open(filename) as f:
        for line in tqdm(f):
            temp = line.strip().split()

            data[count, 0] = float(temp[0])
            data[count, 1] = float(temp[1])
            data[count, 2] = float(temp[2])
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
                label[count, 10] = float(temp[13])
                if count%length==0:
                    name.append(temp[14])
            
            elif (fault_type ==10):
                label[count, 9] = float(temp[12]) 
                if count%length==0:
                    name.append(temp[13])

            count = count + 1
    print (str(count) + 'lines are read, '+ str(length * kinds) +'is respected' )
    return [data, label,name]

from pyhht.visualization import plot_imfs
from pyhht.emd import EMD
import numpy as np
import os
from scipy.signal import hilbert
import matplotlib.pyplot as plt
if_imf = 0
import sys

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)  


phaseA = open("./fanxi/phaseA.txt",'w');
phaseB = open("./fanxi/phaseB.txt",'w');
phaseC = open("./fanxi/phaseC.txt",'w');
filename = './data/313train_data_1600point_randtime1000.txt'
count = 0
name = list()
data = np.zeros((1600, 3))
thezero = np.zeros((1600,1))
with open(filename) as f:
    lines = f.readlines()
    for line in lines:
        if '313_AG_41_0.1585_line313rand1000_0' in line:
            temp = line.strip().split()
        
            data[count, 0] = float(temp[0])
            data[count, 1] = float(temp[1])
            data[count, 2] = float(temp[2])
            
            if count%1600 == 0:
                name.append(temp[14])
            
            count +=1
print(name)

t = np.linspace(0, 1600, 1600, endpoint = False)



if if_imf:
    plt.figure(figsize=(16, 12))
    x = data[0: 1600,0]
    plt.plot(t, data[0: 1600, 0], 'y')
    #decomposer = EMD(x)
    #imfs = decomposer.decompose()
    #plot_imfs("imfs",x, imfs, t)
    plt.show()

else:
    plt.figure(figsize=(16, 12))
    x = data[0: 1600,0]
    # SAVE DATA
    for xx in range(1600):
      print(x[xx],file=phaseA)
    phaseA.close()
    decomposer = EMD(x)
    imfs = decomposer.decompose()

    plt.subplot(3,5,1) # original signal
    plt.plot(t, data[0: 1600, 0], 'y')
    plt.grid()


    hs0 = hilbert(imfs[0])
    hs1 = hilbert(imfs[1])
      

    plt.subplot(3,5,2)
    plt.plot(t, np.abs(hs0), 'b') # amp of imf1
    plt.grid()
    plt.subplot(3,5,3)
    anal_extend = np.append(0, (hs0))
    iphase = np.unwrap(np.angle(anal_extend))
    dt=1.0
    ifreq = np.diff(iphase) / dt
    plt.plot(t, ifreq, 'b')


    plt.subplot(3,5,4)
    plt.plot(t, np.abs(hs1), 'b')
    plt.grid()
    plt.subplot(3,5,5)
    anal_extend = np.append(0, (hs1))
    iphase = np.unwrap(np.angle(anal_extend))
    ifreq = np.diff(iphase) / dt
    plt.plot(t, ifreq, 'b')

    ############################
    
    x = data[0: 1600,1]
    # SAVE DATA
    for xx in range(1600):
      print(x[xx],file=phaseB)
    phaseB.close()
    decomposer = EMD(x)
    imfs = decomposer.decompose()

    plt.subplot(3,5,6)
    plt.plot(t, data[0: 1600, 1], 'y')
    plt.grid()
    hs0 = hilbert(imfs[0])
    if (len(imfs)>1):
        hs1 = hilbert(imfs[1])


    plt.subplot(3,5,7)
    plt.plot(t, np.abs(hs0), 'b')
    plt.grid()
    plt.subplot(3,5,8)
    anal_extend = np.append(0,(hs0))
    iphase = np.unwrap(np.angle(anal_extend))
    dt=1.0
    ifreq = np.diff(iphase) / dt
    plt.plot(t, ifreq, 'b')

    
    plt.subplot(3,5,9)
    if (len(imfs)>1):
        plt.plot(t, np.abs(hs1), 'b')
    else:
        plt.plot(t, thezero, 'b')
    plt.grid()
    plt.subplot(3,5,10)
    anal_extend = np.append(0, (hs1))
    iphase = np.unwrap(np.angle(anal_extend))
    ifreq = np.diff(iphase) / dt
    if (len(imfs)>1):
        plt.plot(t, ifreq, 'b')
    else:
        plt.plot(t, thezero, 'b')
    

    ###########################
    x = data[0: 1600,2]
    # SAVE DATA
    for xx in range(1600):
      print(x[xx],file=phaseC)
    phaseC.close()

    decomposer = EMD(x)
    imfs = decomposer.decompose()

    plt.subplot(3,5,11)
    plt.plot(t, data[0: 1600, 2], 'y')
    plt.grid()
    hs0 = hilbert(imfs[0])
    if (len(imfs)>1):
        hs1 = hilbert(imfs[1])


    plt.subplot(3,5,12)
    plt.plot(t, np.abs(hs0), 'b')
    plt.grid()
    plt.subplot(3,5,13)
    anal_extend = np.append(0, (hs0))
    iphase = np.unwrap(np.angle(anal_extend))
    dt=1.0
    ifreq = np.diff(iphase) / dt
    plt.plot(t, ifreq, 'b')

    
    plt.subplot(3,5,14)
    if (len(imfs)>1):
        plt.plot(t, np.abs(hs1), 'b')
    else:
        plt.plot(t, thezero, 'b')

    plt.grid()
    plt.subplot(3,5,15)
    anal_extend = np.append(0, (hs1))
    iphase = np.unwrap(np.angle(anal_extend))
    ifreq = np.diff(iphase) / dt
    if (len(imfs)>1):
        plt.plot(t, ifreq, 'b')
    else:
        plt.plot(t, thezero, 'b')

    plt.savefig("hs.png")
    plt.show()
    #############noise################

noise = np.zeros((1600, 3))
for i in range(3):
    noise[:, i] = wgn(data[:, i], 40)

data = data + noise

if if_imf:
    x = data[0: 1600,0]
    decomposer = EMD(x)
    imfs = decomposer.decompose()
    plot_imfs("imfs_noise",x, imfs, t)

else:
    plt.figure(figsize=(16, 12))
    
    x = data[0: 1600,0]
    decomposer = EMD(x)
    imfs = decomposer.decompose()

    plt.subplot(3,5,1) # original signal
    plt.plot(t, data[0: 1600, 0], 'y')
    plt.grid()

    hs0 = hilbert(imfs[0])
    hs1 = hilbert(imfs[1])
    for xx in range(1600):
        print (imfs[0][xx])

    plt.subplot(3,5,2)
    plt.plot(t, np.abs(hs0), 'b') # amp of imf1
    plt.grid()
    
    plt.subplot(3,5,3)
    anal_extend = np.append(0, (hs0))
    iphase = np.unwrap(np.angle(anal_extend))
    dt=1.0
    ifreq = np.diff(iphase) / dt
    plt.plot(t, ifreq, 'b')


    plt.subplot(3,5,4)
    plt.plot(t, np.abs(hs1), 'b')
    plt.grid()
    plt.subplot(3,5,5)
    anal_extend = np.append(0, (hs1))
    iphase = np.unwrap(np.angle(anal_extend))
    ifreq = np.diff(iphase) / dt
    plt.plot(t, ifreq, 'b')

    ############################
    
    x = data[0: 1600,1]
    decomposer = EMD(x)
    imfs = decomposer.decompose()

    plt.subplot(3,5,6)
    plt.plot(t, data[0: 1600, 1], 'y')
    plt.grid()
    hs0 = hilbert(imfs[0])
    if (len(imfs)>1):
        hs1 = hilbert(imfs[1])


    plt.subplot(3,5,7)
    plt.plot(t, np.abs(hs0), 'b')
    plt.grid()
    plt.subplot(3,5,8)
    anal_extend = np.append(0,(hs0))
    iphase = np.unwrap(np.angle(anal_extend))
    dt=1.0
    ifreq = np.diff(iphase) / dt
    plt.plot(t, ifreq, 'b')

    
    plt.subplot(3,5,9)
    if (len(imfs)>1):
        plt.plot(t, np.abs(hs1), 'b')
    else:
        plt.plot(t, thezero, 'b')
    plt.grid()
    plt.subplot(3,5,10)
    anal_extend = np.append(0, (hs1))
    iphase = np.unwrap(np.angle(anal_extend))
    ifreq = np.diff(iphase) / dt
    if (len(imfs)>1):
        plt.plot(t, ifreq, 'b')
    else:
        plt.plot(t, thezero, 'b')
    

    ###########################
    x = data[0: 1600,2]
    decomposer = EMD(x)
    imfs = decomposer.decompose()

    plt.subplot(3,5,11)
    plt.plot(t, data[0: 1600, 2], 'y')
    plt.grid()
    hs0 = hilbert(imfs[0])
    if (len(imfs)>1):
        hs1 = hilbert(imfs[1])


    plt.subplot(3,5,12)
    plt.plot(t, np.abs(hs0), 'b')
    plt.grid()
    plt.subplot(3,5,13)
    anal_extend = np.append(0, (hs0))
    iphase = np.unwrap(np.angle(anal_extend))
    dt=1.0
    ifreq = np.diff(iphase) / dt
    plt.plot(t, ifreq, 'b')

    
    plt.subplot(3,5,14)
    if (len(imfs)>1):
        plt.plot(t, np.abs(hs1), 'b')
    else:
        plt.plot(t, thezero, 'b')

    plt.grid()
    plt.subplot(3,5,15)
    anal_extend = np.append(0, (hs1))
    iphase = np.unwrap(np.angle(anal_extend))
    ifreq = np.diff(iphase) / dt
    if (len(imfs)>1):
        plt.plot(t, ifreq, 'b')
    else:
        plt.plot(t, thezero, 'b')

    plt.savefig("hs_noise.png")
    plt.show()

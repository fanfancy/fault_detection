import pywt
import scipy.stats
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

def dwtprocesswhole(data,kinds):
    wavelet = ['db2','sym2','db4','sym4','db6','sym6','db8','sym8','db10']

    #determine the length
    lengthofdwt = 0
    for wave in wavelet:
        core = pywt.Wavelet(wave)
        res = pywt.wavedec(data[0,:,0],core)
        lengthofdwt += len(res)
    lengthofdwt = lengthofdwt*3*6

    input_dwt = np.zeros((kinds,lengthofdwt))  #200

    for num in tqdm(range(kinds)):
        loc = 0
        for phase in range(3):

            for wave in wavelet:

                core = pywt.Wavelet(wave)
                res = pywt.wavedec(data[num,:,phase],core)
                result = np.zeros((len(res)*6))

                for i in range(len(res)):
                    result[i*6+0] = np.max(res[i])
                    result[i*6+1] = np.min(res[i])
                    result[i*6+2] = np.mean(res[i])
                    result[i*6+3] = np.std(res[i])
                    result[i*6+4] = scipy.stats.skew(res[i])
                    result[i*6+5] = np.sum(np.square(res[i]))

                input_dwt[num,loc:loc+len(res)*6] = result
                loc += len(res)*6
    return input_dwt,lengthofdwt

def dwtprocess(data):
    wavelet = ['db2','sym2','db4','sym4','db6','sym6','db8','sym8','db10']

    #determine the length
    lengthofdwt = 0
    for wave in wavelet:
        core = pywt.Wavelet(wave)
        res = pywt.wavedec(data[:,0],core)
        lengthofdwt += len(res)
    lengthofdwt = lengthofdwt*3*6
    input_dwt = np.zeros((lengthofdwt))  #200
    loc = 0
    for phase in range(3):
        for wave in wavelet:
            core = pywt.Wavelet(wave)
            res = pywt.wavedec(data[:,phase],core)
            result = np.zeros((len(res)*6))
            for i in range(len(res)):
                result[i*6+0] = np.max(res[i])
                result[i*6+1] = np.min(res[i])
                result[i*6+2] = np.mean(res[i])
                result[i*6+3] = np.std(res[i])
                result[i*6+4] = scipy.stats.skew(res[i])
                result[i*6+5] = np.sum(np.square(res[i]))
            input_dwt[loc:loc+len(res)*6] = result
            loc += len(res)*6
    return input_dwt,lengthofdwt



def multicore(data, kinds):
    processes = mp.cpu_count()
    pool = mp.Pool(processes=processes)
    tasks = [data[i, :, :] for i in (range(kinds))]

    print (len(tasks))
    #     tasks = [(data[i,:,:],n_imfs) for i in (range(2))]
    res = pool.map(dwtprocess, tqdm(tasks))
    return res
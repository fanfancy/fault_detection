import numpy as np
import utils
import multiprocessing as mp
from tqdm import tqdm
import sys

from scipy.signal import hilbert
from utils import boundary_conditions
from utils import extr
from scipy.signal import argrelmin, argrelmax
from scipy.interpolate import splrep, splev
from pyhht.emd import EMD
from matplotlib import pyplot as plt 


tol = 0.05


def HilTrans(imf, dt=1.0):
    anal = hilbert(imf)
    amp_env = np.abs(anal)
    anal_extend = np.append(0, anal)
    iphase = np.unwrap(np.angle(anal_extend)) 
    ifreq = np.diff(iphase) / dt 
    #ifreq[0:30] = 0
    #ifreq[1570:1600]=0
    #plt.figure()
    #plt.plot(ifreq)
    #plt.savefig("./figure_res/ifreq.png")
    #sys.exit()
    return amp_env, ifreq


def ndiff_extrema_zcrossing(x):
    """Get the difference between the number of zero crossings and extrema."""
    n_max = argrelmax(x)[0].shape[0]
    n_min = argrelmin(x)[0].shape[0]
    n_zeros = (x[:-1] * x[1:] < 0).sum()
    return abs((n_max + n_min) - n_zeros)


def sift(x, ts):
    """One sifting iteration."""
    tmin, tmax, xmin, xmax = boundary_conditions(x, ts)
    tck = splrep(tmin, xmin)
    lower_envelop = splev(ts, tck)
    tck = splrep(tmax, xmax)
    upper_envelop = splev(ts, tck)
    mean_amplitude = np.abs(upper_envelop - lower_envelop) / 2
    local_mean = (upper_envelop + lower_envelop) / 2
    amplitude_error = np.abs(local_mean) / mean_amplitude
    return x - local_mean, amplitude_error.sum()


def emd(x, ts, n_imfs):
    
    x.dtype = 'float64' 
    imfs = np.zeros((n_imfs + 1, x.shape[0]))
    decomposer = EMD(x)
    allimfs = decomposer.decompose()
    if (len(allimfs)>0):    
        imfs[0,:]= allimfs[0]
    else:
        imfs[0,:]= x - imfs.sum(0)

    if (len(allimfs)>1):
        imfs[1,:] = allimfs[1]
    else:
        imfs[1,:] = np.zeros((1,x.shape[0]))

    imfs[-1,:] = x - imfs.sum(0)
    '''
    imfs = np.zeros((n_imfs + 1, x.shape[0]))
    for i in range(n_imfs):
        ix = 0
        #print ("x",x.shape)
        #print ("sum",imfs.sum(0).shape)
        mode = x - imfs.sum(0)
        nmin, nmax, nzero = map(len, extr(mode))
        tmin, tmax, xmin, xmax = boundary_conditions(mode, ts)
        #             while abs((nmin + nmax) - nzero) > 1 and (not (utils.judge_stop(mode))):
        #                 mode, amplitude_error = sift(mode, ts)
        #                 if amplitude_error <= tol or (utils.judge_stop(mode)):
        #                     break
        if abs((nmin + nmax) - nzero) > 1 and (not (utils.judge_stop(mode))) and len(tmin)>3 and len(tmax)>3:
            mode, amplitude_error = sift(mode, ts)
        imfs[i, :] = mode
    imfs[-1, :] = x - imfs.sum(0)
    '''
    return imfs


def hht(data, n_imfs):
    # ====================== Get the imfs of the signal ======================#
    seq_len = data.shape[0]
    num_phases = data.shape[1]
    ts = np.array(range(seq_len))

    if n_imfs == 0:
        imfs = np.reshape(data, [seq_len, 1, num_phases]) 
    else:
        imfs = np.zeros((seq_len, n_imfs, num_phases)) # 1600,3,3
        for i in range(num_phases):
            #print (data[:, i]) #xif
            imfs[:, :, i] = np.transpose(emd(data[:, i], ts, n_imfs=n_imfs)[:n_imfs, :])

    if n_imfs == 0:
        data_after_HHT = np.zeros((seq_len, 2, num_phases))
        for i in range(num_phases):
            for j in range(1):
                data_after_HHT[:, 2 * j, i], data_after_HHT[:, 2 * j + 1, i] = HilTrans(imfs[:, j, i])
    else:
        data_after_HHT = np.zeros((seq_len, n_imfs * 2, num_phases))
        for i in range(num_phases):
            for j in range(n_imfs): # 0,1,2
                data_after_HHT[:, 2 * j, i], data_after_HHT[:, 2 * j + 1, i] = HilTrans(imfs[:, j, i])
    # print (data_after_HHT.shape) # 1600,6,3
    return data_after_HHT

def multicore(data, n_imfs, length):
    processes = mp.cpu_count()

    pool = mp.Pool(processes=processes)
    tasks = [(data[i, :, :], n_imfs) for i in (range(length))]
    #     tasks = [(data[i,:,:],n_imfs) for i in (range(2))]
    res = pool.starmap(hht, tqdm(tasks))
    return res



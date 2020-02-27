from numpy import *
import scipy.io as sio
from clean import *
frame_gen = sio.loadmat('frame_gen.mat')['frame_gen']  # transmit signal template
load_r = sio.loadmat('../data/r1.mat')
r1 = load_r['radar1']  # raw data from the radar

cleanrslt = []
data = np.ones((600, 751))
ycl = 0
leiji = np.zeros((10, 690))

for i in range(0, len(r1)):
    save = r1[i, :]
    data[i, :] = save
    tmp = save[49:-12]  # remove direct path and timestamp
    if tmp != []:
        xx = leiji[0:-1, :]
        leiji[1:len(leiji), ] = xx
        leiji[0, :] = tmp
        if ycl >= 10:
            if ycl == 10:
                print('innit')
            else:
                leiji = leiji[0:10, :]
                meandata = np.mean(leiji, axis=0)
                dedata = leiji[0, :] - meandata  # DC removal
                sig = dedata
                [cmap, dmap, n, ht] = clean(sig, frame_gen, 0.8e-3)  # CLEAN
                ss = ht[0:-25]
                ht[0:25] = np.zeros(25)
                ht[25:] = ss
                cleanrslt.append(ht)
        else:
            None
        ycl = ycl + 1
    else:
        None

sio.savemat('../result/crslt1.mat', {'crslt': cleanrslt})  # output of clean

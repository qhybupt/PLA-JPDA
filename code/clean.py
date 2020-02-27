# CLEAN algorithm
# Input: input_signal，template_signal, Tclean
# Output: ht
import numpy as np


def clean(input_signal, template_signal, Tclean):
    dmap = input_signal  # dirty map
    length = len(input_signal)
    cmap = np.zeros(length)  # clean map
    n = 0  # iteration counter
    tl = template_signal.shape[1]
    tsignal = np.zeros(length)  # zero-padding
    tsignal[0: tl] = template_signal
    ht = np.zeros(length)  # output

    Rxx = np.correlate(tsignal, tsignal)  # auto-correlation
    R0 = Rxx

    Rxy = np.correlate(dmap, tsignal, 'full')  # cross-correlation
    Rl = len(Rxy)
    R = Rxy[Rl//2:] / R0  # normalization

    Rmax = max(R)
    loc = np.argmax(R)

    while Rmax > Tclean:
        maxloc = loc
        ht[maxloc] = Rmax
        stsignal = np.zeros(length)
        if maxloc + tl < length:
            stsignal[maxloc: maxloc + tl] = template_signal
        else:
            stsignal[maxloc:] = template_signal[0, 0:length - maxloc]
        newmap = Rmax * stsignal
        dmap = dmap - newmap
        cmap = cmap + newmap
        n = n + 1
        Rxy = np.correlate(dmap, tsignal, 'full')
        Rl = len(Rxy)
        R = Rxy[Rl // 2:] / R0
        Rmax = max(R)
        loc = np.argmax(R)
    return cmap, dmap, n, ht

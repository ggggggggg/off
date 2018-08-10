import mass
import numpy as np
import pylab as plt
import os
import collections
import off
import scipy as sp
import off_demo
import glob


offPattern   = "../20180810/0005/20180810_run0005_chan*.off"
offFiles= glob.glob(offPattern)
offFiles = offFiles[:min(4,len(offFiles))]
data = collections.OrderedDict()
for offFile in offFiles:
    f = off.OFFFile(offFile)
    ch = off_demo.OFFChannel(f)
    try:
        print offFile
        ch.learnAll()
        data[f.header["ChannelNumberMatchingName"]] = ch
    except Exception as ex:
        print "channel {} failed".format(f.header["ChannelNumberMatchingName"])
        print ex

binEdges = np.arange(7000,10000,2)
binCenters = off_demo.midpoints(binEdges)
def comboHist():
    counts = np.zeros_like(binCenters)
    for channelNumber,ch in data.iteritems():
        ch.off._updateMmap()
        counts += ch.hist(binEdges)
    return counts

def plotComboHist():
    counts = comboHist()
    plt.plot(binCenters,counts)
    plt.xlabel("energy (eV)")
    plt.ylabel("counts per bin")

plotComboHist()

import mass
import numpy as np
import pylab as plt
import os
import collections
import off
import scipy as sp

def midpoints(r):
    return 0.5*(r[1:]+r[:-1])


def _find_peaks_heuristic(phnorm):
    """A heuristic method to identify the peaks in a spectrum.

    This can be used to design the arrival-time-bias correction. Of course,
    you might have better luck finding peaks by an experiment-specific
    method, but this will stand in if you cannot or do not want to find
    peaks another way.

    Args:
        phnorm: a vector of pulse heights, found by whatever means you like.
            Normally it will be the self.p_filt_value_dc AFTER CUTS.

    Returns:
        ndarray of the various peaks found in the input vector.
    """
    median_scale = np.median(phnorm)

    # First make histogram with bins = 0.2% of median PH
    hist, bins = np.histogram(phnorm, 1000, [0, 2*median_scale])
    binctr = bins[1:] - 0.5 * (bins[1] - bins[0])

    # Scipy continuous wavelet transform
    pk1 = np.array(sp.signal.find_peaks_cwt(hist, np.array([2, 4, 8, 12])))

    # A peak must contain 0.5% of the data or 500 events, whichever is more,
    # but the requirement is not more than 5% of data (for meager data sets)
    Ntotal = len(phnorm)
    MinCountsInPeak = min(max(500, Ntotal//200), Ntotal//20)
    pk2 = pk1[hist[pk1] > MinCountsInPeak]

    # Now take peaks from highest to lowest, provided they are at least 40 bins from any neighbor
    ordering = hist[pk2].argsort()
    pk2 = pk2[ordering]
    peaks = [pk2[0]]

    for pk in pk2[1:]:
        if (np.abs(peaks-pk) > 10).all():
            peaks.append(pk)
    peaks.sort()
    return np.array(binctr[peaks])

class OFFChannel():
    def __init__(self,offIn,ljh=None):
        self.off=offIn # an off.OFFFile
        self.ljh=ljh

    def plotTrace(self,i):
        plt.figure()
        x,y = f.recordXY(i)
        plt.plot(x*1e3,y,label="off record {}".format(i))
        if ljh is not None:
            yljh = ljh[i]
        plt.plot(x*1e3,yljh,label="ljh record {}".format(i))
        plt.title(self.shortname)
        plt.legend(loc="best")
        plt.xlabel("time (ms)")
        plt.ylabel("signal")

    @property
    def shortname(self):
        return f.filename.split("/")[-1]

    @property
    def fv(self):
        # assumees coefs go [mean, deriv, average pulse, other...]
        return self.off["coefs"][:,2]

    @property
    def phase(self):
        # assumees coefs go [mean, deriv, average pulse, other...]
        return self.off["coefs"][:,1]/self.dc[:]/np.dot(self.off.basis[2:,2],self.off.basis[1:-1,1])

    @property
    def pretriggerMean(self):
        return self.off["pretriggerMean"]

    @property
    def good(self):
        # cut pulses with
        # 1. residualStdDev too large
        # 2. firstSampleMinusPretrigMean too large
        firstSample = np.matmul(self.off.basis[0:1,:],self.off["coefs"].T)[0]
        firstSampleMinusPretrigMean = (firstSample-self.off["pretriggerMean"])
        g = np.logical_and(self.off["residualStdDev"]/np.median(self.off["residualStdDev"]) < 3,
                           np.abs(firstSampleMinusPretrigMean)<np.median(self.off["residualStdDev"])*3)
        return g

    def learnDriftCorrection(self):
        indicator = self.pretriggerMean[self.good]
        uncorrected = self.fv[self.good]
        drift_corr_param, drift_correct_info = mass.core.analysis_algorithms.drift_correct(indicator, uncorrected)
        ptm_offset = drift_correct_info["median_pretrig_mean"]
        slope = drift_correct_info["slope"]
        assert drift_correct_info["type"] == "ptmean_gain"
        self.dcOffset = ptm_offset
        self.dcSlope = slope

    @property
    def dc(self):
        gain = 1+(self.pretriggerMean-self.dcOffset)*self.dcSlope
        dc = self.fv*gain
        return dc

    def plotDriftCorrection(self):
        plt.figure()
        plt.plot(self.pretriggerMean[self.good], self.fv[self.good],".")
        plt.plot(self.pretriggerMean[self.good], self.dc[self.good],".")
        plt.xlabel("pretriggerMean")
        plt.ylabel("fv and dc")
        plt.title(self.shortname)

    def learnPhaseCorrection(self):
        ph_peaks = _find_peaks_heuristic(self.dc[self.good])
        if len(ph_peaks) <= 0:
            raise ValueError("Could not phase_correct on chan %3d because no peaks"%f.header["ChannelNumberMatchingName"])
        ph_peaks = np.asarray(ph_peaks)
        ph_peaks.sort()

        # Compute a correction function at each line in ph_peaks
        corrections = []
        median_phase = []
        kernel_width = np.max(ph_peaks)/1000.0
        for pk in ph_peaks:
            c, mphase = mass.channel._phasecorr_find_alignment(self.phase[self.good],
                                                  self.dc[self.good], pk,
                                                  .012*np.mean(ph_peaks),
                                                  method2017=True,
                                                  kernel_width=kernel_width)
            corrections.append(c)
            median_phase.append(mphase)
        median_phase = np.array(median_phase)
        self.phaseCorrections = corrections

    @property
    def phc(self):
        return mass.channel._phase_corrected_filtvals(self.phase, self.dc, self.phaseCorrections)

    def plotPhaseCorrection(self):
        plt.plot(self.phase[self.good], self.dc[self.good],".",label="dc")
        plt.plot(self.phase[self.good], self.phc[self.good],".",label="phc")
        plt.xlabel("phase")
        plt.ylabel("dc and phc")
        plt.legend()
        plt.title(self.shortname)

    def learnCalibration(self):
        auto_cal = mass.EnergyCalibrationAutocal(mass.EnergyCalibration(),
                                            self.phc[self.good],
                                            ["CuKAlpha","CuKBeta"])
        auto_cal.guess_fit_params(smoothing_res_ph=100,
                                  fit_range_ev=100,
                                  binsize_ev=1,
                                  nextra=0, maxacc=0.015)
        auto_cal.fit_lines()
        self.auto_cal = auto_cal

    def plotCalibration(self):
        self.auto_cal.diagnose()

    @property
    def energy(self):
        return self.auto_cal.calibration(self.phc)

    def hist(self, binEdges=np.arange(7000,10000,2)):
        counts, _ = np.histogram(self.energy[self.good],binEdges)
        return counts

    def plotHist(self, binEdges=np.arange(7000,10000,2)):
        counts = self.hist(binEdges)
        x = midpoints(binEdges)
        plt.figure()
        plt.plot(x,counts)
        plt.xlabel("energy (eV)")
        plt.ylabel("counts per bin")
        plt.title(self.shortname)

    def learnAll(self):
        self.learnDriftCorrection()
        self.learnPhaseCorrection()
        self.learnCalibration()


if __name__ == "__main__":
    noise_file = "../20180810/0003/20180810_run0003_chan1.ljh"
    pulse_file = "../20180810/0005/20180810_run0005_chan1.ljh"
    off_file   = "../20180810/0005/20180810_run0005_chan1.off"

    outdir = os.path.split(os.path.dirname(pulse_file))[-1] # pull out the final directory name, eg 0039 from "../20180810/0039/20180810_run0039_chan1.ljh"
    if not os.path.isdir(outdir):
        os.mkdir(outdir)



    f = off.OFFFile(off_file)
    ch = OFFChannel(f)
    ch.learnAll()
    ch.plotCalibration()
    ch.plotHist()

import mass
import off
import h5py
import numpy as np
import json
import base64


ljhfile = "../20180801/0040/20180801_run0040_chan1.ljh"
projectorsfile = "../20180801/0039/20180801_run0039_model.hdf5"
offfile = ljhfile[:-3]+"off_frompy"

ljh = mass.LJHFile(ljhfile)
projectorsh5 = h5py.File(projectorsfile)
chanGroup = projectorsh5["{}".format(ljh.channum)]
projectors = np.array(chanGroup["svdbasis"]["projectors"].value.T, np.float64)
basis = np.array(chanGroup["svdbasis"]["basis"].value.T, np.float64)
modelCoefs = np.array(np.matmul(projectors,ljh[0]),dtype=np.float32)
nbasis = projectors.shape[0]

offVersion = "0.1.0"
offHeader = {
    "FileFormatVersion": offVersion,
    "FramePeriodSeconds":ljh.timebase,
    "NumberOfBases":projectors.shape[0],
    "FileFormat":"OFF"
}
offHeader["ModelInfo"]={
    "Projectors":{
        "RowMajorFloat64ValuesBase64":base64.b64encode(projectors.copy(order="C")),
        "Rows":projectors.shape[0],
        "Cols":projectors.shape[1]
},
    "Basis":{
        "RowMajorFloat64ValuesBase64":base64.b64encode(basis.copy(order="C")),
        "Rows":basis.shape[0],
        "Cols":basis.shape[1]
}}
offHeader["ReadoutInfo"] = {
    "ColumnNum": ljh.column_number,
    "RowNum": ljh.row_number,
    "NumberOfColumns": ljh.number_of_columns,
    "NumberOfRows": ljh.number_of_rows,
}
offHeader["CreationInfo"]={
    "SourceName": "ljh2off.py"
}
headerString = json.dumps(offHeader,indent=4,sort_keys=True)

dtype = off.recordDtype(offVersion, nbasis)

framecounts = ljh.rowcount//ljh.number_of_rows
unixnanos = np.array(np.round(ljh.datatimes_float*1e9),dtype="int")
pretriggerMean = np.mean(ljh[0][:ljh.nPresamples])
residualStdDev = 0
modelCoefs = np.array(np.matmul(projectors,ljh[0]),dtype=np.float32)
modeledPulse = np.matmul(basis,modelCoefs)
residual = modeledPulse-ljh[0]
residualStdDev = np.std(residual)
a=np.array([(ljh.nSamples, ljh.nPresamples, framecounts[0], unixnanos[0],
          pretriggerMean, residualStdDev, modelCoefs)],dtype=dtype)

with open(offfile,"w") as f:
    f.write(headerString)
    f.write("\n")
    for (first, lastplus1, segnum, data) in ljh.iter_segments(first=0,end=-1):
        framecounts = ljh.rowcount//ljh.number_of_rows
        unixnanos = np.array(np.round(ljh.datatimes_float*1e9),dtype="int")
        for i in range(len(ljh.rowcount)):
            pulse = data[i,:]
            pretriggerMean = np.mean(pulse[:ljh.nPresamples])
            modelCoefs = np.array(np.matmul(projectors,pulse),dtype=np.float32)
            modeledPulse = np.matmul(basis,modelCoefs)
            residual = modeledPulse-pulse
            residualStdDev = np.std(residual)
            a=np.array([(ljh.nSamples, ljh.nPresamples, framecounts[i], unixnanos[i],
                  pretriggerMean, residualStdDev, modelCoefs)],dtype=dtype)
            a.tofile(f)

f1 = off.OFFFile(ljhfile[:-3]+"off")
f2 = off.OFFFile(ljhfile[:-3]+"off_frompy")


# julia ~/.julia/v0.6/Pope/scripts/basis_create.jl python_misc/20180801/0039/20180801_run0039_chan1.ljh python_misc/20180801/0038/20180801_run0038_noise.hdf5 --replaceoutput

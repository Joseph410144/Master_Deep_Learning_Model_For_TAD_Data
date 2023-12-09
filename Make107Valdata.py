import os
from mne.io import read_raw_edf
import numpy as np
from scipy import signal
import xml.dom.minidom
from pathlib import Path


def SignalFilter(data, freq, lf, hf):
    """filter"""
    ln = (2*lf)/freq
    hn = (2*hf)/freq
    b, a = signal.butter(4, [ln, hn], "bandpass")
    filteddata = signal.filtfilt(b, a, data)
    return filteddata

def DataProcessing(rawdata, freq, resample_freq):
    eeg = (rawdata.T).squeeze()
    eeg = np.array(eeg)
    if freq >= 200:
        filteddata = SignalFilter(eeg, freq, 0.5, 60)
    else:
        filteddata = eeg
    # all sec
    AllSec = len(filteddata)//freq
    ReSampleData = signal.resample(filteddata, AllSec*resample_freq)
    return ReSampleData

def zscore_normalize_features(X):
    mu     = np.mean(X)                 
    sigma  = np.std(X)               
    X_norm = (X - mu) / sigma
    return (X_norm, mu, sigma)


def MakeValData(dataPath):

    path_patient = dataPath
    path_patient_content = os.listdir(path_patient)
    path_xml = os.path.join(path_patient, path_patient_content[-1])
    path_patient_content_inside = os.listdir(os.path.join(path_patient, path_patient_content[0]))
    path_edf = os.path.join(os.path.join(path_patient, path_patient_content[0]), path_patient_content_inside[-1])
    tempXml = Path(path_xml)
    if not tempXml.is_file():
        return False, False, False
    doc=xml.dom.minidom.parse(path_xml)
    root = doc.documentElement 
    events = root.getElementsByTagName("Event")
    raw=read_raw_edf(path_edf, preload=False)
    EdfTime = str(raw.info["meas_date"]).split(" ")[-1]
    EdfTime = int(EdfTime.split(":")[0])*60*60 + int(EdfTime.split(":")[1])*60 + int(EdfTime.split(":")[2].split("+")[0])
    apneaStarttime = []
    apneaendtime = []
    AllStarttime = events[0].getElementsByTagName("StartTime").item(0).childNodes[0].nodeValue.split("T")[1]
    AllStarttime = str(AllStarttime)
    AllStarttime = int(AllStarttime.split(":")[0])*60*60 + int(AllStarttime.split(":")[1])*60 + int(AllStarttime.split(":")[2].split(".")[0])
    for event in events:
        starttimeNode = event.getElementsByTagName("StartTime").item(0)
        typeNode = event.getElementsByTagName("Type").item(0)
        time = event.getElementsByTagName("StartTime").item(0).childNodes[0].nodeValue.split("T")[1]
        value = typeNode.childNodes[0].nodeValue
        if "APNEA-CENTRAL" in typeNode.childNodes[0].nodeValue or "APNEA-MIXED" in typeNode.childNodes[0].nodeValue or "APNEA-OBSTRUCTIVE" in typeNode.childNodes[0].nodeValue:
            StartapneaTime = event.getElementsByTagName("StartTime").item(0).childNodes[0].nodeValue.split("T")[1]
            EndapneaTime = event.getElementsByTagName("StopTime").item(0).childNodes[0].nodeValue.split("T")[1]
            StartapneaTime = str(StartapneaTime)
            EndapneaTime = str(EndapneaTime)
            if int(StartapneaTime.split(":")[0])>=0 and int(StartapneaTime.split(":")[0]) <= 9:
                StartapneaTime = (int(StartapneaTime.split(":")[0])+24)*60*60 + int(StartapneaTime.split(":")[1])*60 + int(StartapneaTime.split(":")[2].split(".")[0])
            else:
                StartapneaTime = int(StartapneaTime.split(":")[0])*60*60 + int(StartapneaTime.split(":")[1])*60 + int(StartapneaTime.split(":")[2].split(".")[0])
            if int(EndapneaTime.split(":")[0])>=0 and int(EndapneaTime.split(":")[0]) <= 9:
                EndapneaTime = (int(EndapneaTime.split(":")[0])+24)*60*60 + int(EndapneaTime.split(":")[1])*60 + int(EndapneaTime.split(":")[2].split(".")[0])
            else:
                EndapneaTime = int(EndapneaTime.split(":")[0])*60*60 + int(EndapneaTime.split(":")[1])*60 + int(EndapneaTime.split(":")[2].split(".")[0])
            apneaStarttime.append(StartapneaTime-EdfTime)
            apneaendtime.append(EndapneaTime-EdfTime)

    raw=read_raw_edf(path_edf, preload=False)
    ChannelsName = list(raw.ch_names)

    """ in mne, if not the same sample rate will get assert error """
    temp = list(raw.ch_names)
    temp.remove("Abdomen")
    rawabd=read_raw_edf(path_edf, preload=False, exclude=tuple(temp))

    temp = list(raw.ch_names)
    temp.remove("SpO2")
    rawsp=read_raw_edf(path_edf, preload=False, exclude=tuple(temp))

    

    """EKG data"""
    temp = ChannelsName.copy()
    temp.remove("EKG")
    raw=read_raw_edf(path_edf, preload=False, exclude=tuple(temp))
    freq = int(raw.info['sfreq'])
    eeg, time = raw[0,:]
    CheckLength = len((eeg.T).squeeze())
    resample_freq = 100
    Ekg = DataProcessing(eeg, freq, resample_freq)
    
    """Arousal label data"""
    Labeldata = np.zeros(len(Ekg))
    for i in range(0, len(apneaStarttime)):
        Labeldata[apneaStarttime[i]*resample_freq:apneaendtime[i]*resample_freq] = 1

    """Nasal Pressure"""
    temp = ChannelsName.copy()
    temp.remove("Nasal Pressure")
    raw=read_raw_edf(path_edf, preload=False, exclude=tuple(temp))
    freq = int(raw.info['sfreq'])
    eeg, time = raw[0,:]
    CheckLength = len((eeg.T).squeeze())
    Nasal = DataProcessing(eeg, freq, resample_freq)
    
    """Abdomen data"""
    eeg, time = rawabd[0,:]
    freqabd = int(rawabd.info['sfreq'])
    Abdomen = DataProcessing(eeg, freqabd, resample_freq)

    """SpO2"""
    eeg, time = rawsp[0,:]
    freqsp = int(rawsp.info['sfreq'])
    Sp = DataProcessing(eeg, freqsp, resample_freq)

    """Chin data"""
    temp = ChannelsName.copy()
    temp.remove("ChinL")
    raw=read_raw_edf(path_edf, preload=False, exclude=tuple(temp))
    freq = int(raw.info['sfreq'])
    eeg, time = raw[0,:]
    temp = ChannelsName.copy()
    temp.remove("ChinR")
    raw=read_raw_edf(path_edf, preload=False, exclude=tuple(temp))
    freq = int(raw.info['sfreq'])
    emg, time = raw[0,:]
    emgL = DataProcessing(eeg, freq, resample_freq)
    emgR = DataProcessing(emg, freq, resample_freq)
    Chin = emgR-emgL
    EpochLength = 5*60*1
    Allepoch = (len(Chin)//resample_freq)//EpochLength
    allTraindata = np.array([zscore_normalize_features(Nasal[:Allepoch*resample_freq*EpochLength])[0], zscore_normalize_features(Abdomen[:Allepoch*resample_freq*EpochLength])[0], 
                                zscore_normalize_features(Chin[:Allepoch*resample_freq*EpochLength])[0], zscore_normalize_features(Ekg[:Allepoch*resample_freq*EpochLength])[0], 
                                zscore_normalize_features(Sp[:Allepoch*resample_freq*EpochLength])[0]])
    Labeldata = np.array([Labeldata])

    return allTraindata, Labeldata, Allepoch


if __name__ == "__main__":
    path = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\AHIClass\only107\Mild"
    a, b = MakeValData(path)
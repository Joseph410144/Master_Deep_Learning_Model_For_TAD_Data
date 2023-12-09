import numpy as np
import os

rawDataPath = r"F:\北醫睡眠中心\睡眠原始檢查資料(EDF+XML)\104"
FolderPath = os.listdir(rawDataPath)
for date in FolderPath:
    path = os.path.join(rawDataPath, date)
    Patients = os.listdir(path)
    for patients in Patients:
        print(os.listdir(os.path.join(path, patients)))

 

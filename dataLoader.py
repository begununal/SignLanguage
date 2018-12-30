import os
import cv2
import glob
import matplotlib
from matplotlib import pyplot as plt


dataSetPaths = ['/dark/Databases/HandShapeClassification/ASL_FINGERSPELLING/ds5',
                '/dark/Databases/HandShapeClassification/ASL_FINGERSPELLINGds9',
                '/dark/Databases/HandShapeClassification/HGR/original_images',
                '/dark/Databases/HandShapeClassification/LSA16']

dataSetNames = ['ds5','ds9','hgr','lsa16']

ImageList = []
LabelList = []
UserList = []
DatasetNameList = []


def readImageDatasets():
    for ds in range(0,len(dataSetNames)):
        if dataSetNames[ds] == 'ds5':
            userNames = sorted(os.listdir(dataSetPaths[ds]))
            for un in range(0, len(userNames)):
                classNames = sorted(os.listdir(dataSetPaths[ds] + os.sep + userNames[un]))
                for cn in range(0, len(classNames)):
                    sampleNames = sorted(glob.glob(dataSetPaths[ds] + os.sep + userNames[un] + os.sep + classNames[cn]+ os.sep + 'color*'))
                    cnt = 0
                    for sn in range(0, len(sampleNames)):
                        imagePath  = sampleNames[sn]
                        img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),(32,32))
                        ImageList.append(img)
                        LabelList.append(classNames[cn])
                        UserList.append(userNames[un])
                        DatasetNameList.append(dataSetNames[ds])
                        print('Read :' + format(cnt) + ' '+ classNames[cn]+' '+ userNames[un]+' '+dataSetNames[ds])
                        cnt = cnt +1

    return ImageList, LabelList, UserList, DatasetNameList

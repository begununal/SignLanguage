import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import numpy as np
import glob
import os.path

def SVM(key, params):
    [train_features, val_features, train_imgs, val_imgs, y_train, y_val] = pickle.load(open("results/Results_" + str(key) + ".pickle", 'rb'))
    shp = train_features[0].shape
    train_features = train_features[0].reshape((shp[0],(shp[1] * shp[2] *shp[3])))
    shp = val_features[0].shape
    val_features = val_features[0].reshape((shp[0],(shp[1] * shp[2] *shp[3])))
    clf = LinearSVC(random_state=0)
    clf.fit(train_features, y_train)
    y_pred = clf.predict(val_features)
    conf = confusion_matrix(y_val, y_pred)
    print(conf)
    confSum = 0
    diagonalSum = 0
    classWiseAcc = []
    res = []
    resDict = {}
    for i in range(0, len(conf)):
        confSum += sum(conf[i])
        diagonalSum += conf[i][i]
        classWiseAcc.append(conf[i][i] / float(sum(conf[i])))

    res.append(params)
    res.append(diagonalSum / float(confSum))
    res.append(classWiseAcc)

    if os.path.isfile("./accuracy.txt"):
        resDict = pickle.load(open("accuracy.txt", "rb"))

    resDict[key] = res
    pickle.dump(resDict, open("accuracy.txt", "wb"))

    print("accuracy : ", diagonalSum/float(confSum))
    print("class-wise: ", classWiseAcc)
    print('debug')
    return diagonalSum/float(confSum), classWiseAcc
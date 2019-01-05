import torch
import numpy as np
import warnings
import heapq
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")
def F_test(classes_n,feature,label,allnum):
    allfeature=feature[0].data.numpy().reshape((feature[0].data.numpy().shape[0],-1))
    for i in range(1,len(feature)):
        feature[i]=feature[i].data.numpy().reshape((feature[i].data.numpy().shape[0],-1))
        allfeature=np.concatenate((allfeature,feature[i]),axis=1)
    #allfeature(60000, 63488)
    allmean=np.mean(allfeature,axis=0)
    splitmeans=[]
    pernum=[]
    for i in range(classes_n):
        oneclassfeature = np.where(label == i)
        one_feature = allfeature[oneclassfeature]
        tmpfeature = np.copy(one_feature)
        pernum.append(tmpfeature.shape[0])
        one_means=np.mean(tmpfeature,axis=0)
        splitmeans.append(one_means)
    splitmeans=np.array(splitmeans)
    BGV=np.zeros(allmean.shape)
    for i in range(classes_n):
        BGV+=pernum[i]*np.square(splitmeans[i]-allmean)/(classes_n-1)
    WGV=np.zeros(tmpfeature[0].shape)
    for i in range(classes_n):
        oneclassfeature = np.where(label == i)
        one_feature = allfeature[oneclassfeature]
        tmpfeature = np.copy(one_feature)
        #print(tmpfeature.shape)
        # (5923, 63488)
        # (6742, 63488)
        # (5958, 63488)
        # (6131, 63488)
        # (5842, 63488)
        # (5421, 63488)
        # (5918, 63488)
        # (6265, 63488)
        # (5851, 63488)
        # (5949, 63488)
        for j in range(pernum[i]):
            WGV+=np.square(tmpfeature[j]-splitmeans[i])/(allnum-classes_n)
    return BGV/WGV
def transdata2mat(feature):
    allfeature = feature[0].data.numpy().reshape((feature[0].data.numpy().shape[0], -1))
    for i in range(1, len(feature)):
        feature[i] = feature[i].data.numpy().reshape((feature[i].data.numpy().shape[0], -1))
        allfeature = np.concatenate((allfeature, feature[i]), axis=1)
    return allfeature
def selectfeature(selindex,feature):
        return feature[:, list(selindex)]

if __name__ == '__main__':
    feature = torch.load('./data/feature')
    allfeature=transdata2mat(feature)
    label = torch.load('./data/label')
    #get the feature's scores
    # F_scores=F_test(classes_n=10,feature=feature,label=label,allnum=60000)
    # F_scores[np.isnan(F_scores)] = 0
    # torch.save(F_scores, './data/F_scores')
    F_scores = list(torch.load('./data/F_scores'))
    #thrid situation
    max_2000_all_list = map(F_scores.index, heapq.nlargest(2000, F_scores))
    # F_last_stage = F_scores[-32768:]
    # max_2000_laststage_list = map(F_last_stage.index, heapq.nlargest(2000, F_last_stage))
    # featurecopy=np.copy(feature)
    # arr1=selectfeature([],featurecopy)
    # featurecopy = np.copy(feature)
    #
    # arr2=selectfeature(max_2000_laststage_list, featurecopy)
    featurecopy = np.copy(allfeature)
    arr3 = selectfeature(max_2000_all_list, featurecopy)
    pca=PCA(n_components=256)#options:64,128,256
    Y=pca.fit_transform(arr3)
    train_x, test_x, train_y, test_y = train_test_split(Y, label, test_size=0.2)
    clf = SVC(kernel='linear', C=0.4)
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)
    print(classification_report(test_y, pred_y))
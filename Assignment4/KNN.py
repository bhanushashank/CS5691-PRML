#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import statistics
from sklearn import svm, datasets
import seaborn as sns
from sklearn.metrics import DetCurveDisplay
import scipy


# ## Helper functions for kNN

# In[2]:


def normalize(x):
    low = min(x)
    high = max(x)
    for i in range(len(x)):
        x[i] -= low
        x[i] /= (high-low)
    return x

def Normalize(train_data, dev_data):
    d = len(train_data[0])
    for i in range(d):
        temp = []
        for j in range(len(train_data)):
            temp.append(train_data[j][i])
        temp.sort()
        Minimum = temp[0]
        Maximum = temp[-1]
        for j in range(len(train_data)):
            train_data[j][i] -= Minimum
            train_data[j][i] /= (Maximum - Minimum)
        for j in range(len(dev_data)):
            dev_data[j][i] -= Minimum
            dev_data[j][i] /= (Maximum - Minimum)
    return train_data,dev_data

def EuclideanDistance(vector1, vector2):
    assert(len(vector1) == len(vector2))
    distance = 0.0
    for i in range(len(vector1)):
        distance += (vector1[i] - vector2[i])**2
    return distance**0.5

def ComputeAccuracy(predicted_class, actual_class):
    assert(len(predicted_class) == len(actual_class))
    match = 0
    total = len(predicted_class)
    for i in range(total):
        if predicted_class[i] == actual_class[i]:
            match += 1
    return match/total

def ComputeDistance(test_point, train_data, train_data_class):
    distances = []
    for i in range(len(train_data)):
        distances.append([EuclideanDistance(train_data[i], test_point), train_data_class[i]])
    distances.sort()
    return distances

def Majority(distances, topK, numClasses):
    assert(len(distances) >= topK)
    classes = [distances[i][1] for i in range(topK)]
    likelihoods = [0 for i in range(numClasses)]
    for i in range(topK):
        likelihoods[int(distances[i][1])] += 1
    for i in range(len(likelihoods)):
        likelihoods[i] /= topK
    return statistics.mode(classes), likelihoods

def kNN(train_data, train_data_class, test_data, K, numClasses):
    assert(len(train_data) == len(train_data_class))
    predicted_class = []
    likelihoods_class = []
    for test_point in test_data:
        distances = ComputeDistance(test_point, train_data, train_data_class)
        prediction, likelihoods = Majority(distances, K, numClasses)
        predicted_class.append(prediction)
        likelihoods_class.append(likelihoods)
    return predicted_class, likelihoods_class

def KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K, numClasses):
    accuracies = []
    predictions = []
    likelihoods = []
    for K_current in K:
        predicted_class, likelihoods_class = kNN(train_data, train_data_class, dev_data, K_current, numClasses)
        accuracy = ComputeAccuracy(predicted_class, dev_data_class)
        accuracies.append(accuracy*100)
        predictions.append(predicted_class)
        likelihoods.append(likelihoods_class)
    return accuracies, predictions, likelihoods


# ## Functions for ROC, DET, Confusion Matrix

# In[3]:


def ROC(likelihood,groundTruth,nClasses):
    S = []
    numberOfTests = len(likelihood)
    for i in range(numberOfTests):
        Total = 0
        for j in range(nClasses):
            Total += likelihood[i][j]
        for j in range(nClasses):
            S.append(likelihood[i][j]/Total)
    
    S.sort()
    threshold = S
    
    Minimum = min(S)
    Maximum = max(S)
    
    #print(Minimum,Maximum)
    
    threshold = np.linspace(Minimum,Maximum,100)
    
    TPR = []
    FPR = []
    FNR = []
    # TPR = TP/(TP+FN) # True positive rate
    # FPR = FP/(FP+TN) # False positive rate
    for i in threshold:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(numberOfTests):
            for k in range(nClasses):
                if likelihood[j][k] >= i:
                    if groundTruth[j] == (k):
                        TP += 1
                    else:
                        FP += 1
                else:
                    if groundTruth[j] == (k):
                        FN += 1
                    else:
                        TN += 1
        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))
        FNR.append(FN / (FN + TP))
    return TPR, FPR, FNR

def PlotROC(log_likelihoods_k, actual_class, numClasses, K_range, plot_title, legend_list):
    count = 0
    TPR_k = []
    FPR_k = []
    FNR_k = []
    for i in range(len(K_range)):
        log_likelihoods = log_likelihoods_k[i]
        TPR, FPR, FNR = ROC(log_likelihoods, actual_class, numClasses)
        TPR_k.append(TPR)
        FPR_k.append(FPR)
        FNR_k.append(FNR)
        plt.plot(FPR, TPR)
        count += 1
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(plot_title)
    plt.legend(legend_list)
    plt.show()
    return TPR_k, FPR_k, FNR_k

def PlotDET(TPR_k, FPR_k, FNR_k, title="DET Curve", legend=""):
    fig, ax_det = plt.subplots(1, 1)
    fig = plt.figure(figsize=(14, 9))
    for i in range(len(TPR_k)):
        FPR = FPR_k[i]
        FNR = FNR_k[i]
        display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)
    ax_det.set_title(title)
    ax_det.set_xlabel('False Positive Rate (FPR)')
    ax_det.set_ylabel('False Negative Rate (FNR)')
    ax_det.legend(legend)
    plt.show()

def ConfusionMatrix(predicted_class, actual_class, classes, title='Confusion Matrix'):
    N = len(predicted_class)
    count = [[0 for i in range(classes)] for j in range(classes)]
    for i in range(N):
        count[int(actual_class[i])][int(predicted_class[i])] += 1
    
    ax = sns.heatmap(count, annot=True)
    ax.set_xlabel('Actual Class', fontsize=16)
    ax.set_ylabel('Predicted Class', fontsize=16)
    ax.set_title(title, fontsize=20)
    plt.show()


# ## Functions for PCA

# In[4]:


def NormalizeData(data):
    N = len(data)
    D = len(data[0])
    for i in range(N):
        data[i] = np.array(data[i])
        
    mean = np.zeros(D)
    for i in range(N):
        mean = mean + data[i]
    mean = mean/N
    for i in range(N):
        data[i] = data[i] - mean
        data[i] = list(data[i])
    return data

def EstimateCovariance(data):
    D = len(data[0])
    N = len(data)
    covariance = np.zeros((D, D))
    
    for i in range(len(data)):
        current_vector = np.reshape(data[i], (D, 1))
        product = current_vector @ current_vector.T
        covariance += product/N
    return covariance

def mag(x, y):
    """Calulates the magnitude of complex number x + jy"""
    return (x*x + y*y)**0.5

def SortMatrix(M, lst):
    """Sorts lst in decreasing order and rearranges columns of M correspondingly"""
    n = M.shape[1]
    if n != len(lst):
        print("Invalid arguments")
        return M, lst
    lst_mag = [mag(x.real, x.imag) for x in lst]
    lst_mag = np.array(lst_mag)
    sorted_indices = lst_mag.argsort()[::-1]
    lst = lst[sorted_indices]
    M = M[:, sorted_indices]
    return M, lst

def ConvertToReal(train_data, dev_data):
    train_data_real = [[] for i in range(len(train_data))]
    dev_data_real = [[] for i in range(len(dev_data))]
    
    for i in range(len(train_data)):
        for j in range(len(train_data[0])):
            train_data_real[i].append(float(train_data[i][j].real))
    for i in range(len(dev_data)):
        for j in range(len(dev_data[0])):
            dev_data_real[i].append(float(dev_data[i][j].real))
    return train_data_real, dev_data_real

def PCA(data, K):
    #normalized_data = NormalizeData(data)
    normalized_data = data
    covariance_matrix = EstimateCovariance(normalized_data)
    
    eigenValues, eigenVectors = np.linalg.eig(covariance_matrix)
    eigenVectors, eigenValues = SortMatrix(eigenVectors, eigenValues)
    eigenVectors_reduced = np.array(eigenVectors[:,:K])
    data = np.array(data)
    data_reduced = data @ eigenVectors_reduced
    return data_reduced, eigenVectors_reduced

def ProjectData(data, principal_components):
    projected_data = np.array(data)
    projected_data = projected_data @ principal_components
    return projected_data

def ReduceDimensions_PCA(train_data, dev_data, ndims):
    train_data_reduced, eigenVectors_reduced = PCA(train_data, ndims)
    dev_data_reduced = ProjectData(dev_data, eigenVectors_reduced)
    train_data_reduced, dev_data_reduced = ConvertToReal(train_data_reduced, dev_data_reduced)
    return train_data_reduced, dev_data_reduced

def ComputeAccuracies_PCA_KNN(train_data, dev_data, K, ndims):
    train_data_reduced, dev_data_reduced = ReduceDimensions_PCA(train_data, dev_data, ndims)
    accuracies = KNN_Classifier(train_data_reduced, train_data_class, dev_data_reduced, dev_data_class, K) 
    return accuracies


# ## Functions for LDA

# In[5]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def ComputeCovariance(X):
    d = len(X[0])
    mean = [0 for i in range(d)]
    for i in range(len(X)):
        mean = mean + X[0]
    mean /= len(X)
    
    covariance = np.zeros((d,d))
    for i in range(len(X)):
        temp = X[0] - mean
        covariance += (temp.T) @ temp
        
    covariance /= len(X)
    return covariance

def Seperation(x_train,c_train):
    Final_train = []
    nClasses = len(set(c_train))
    for i in range(nClasses):
        Final_train.append([])
    for i in range(len(x_train)):
        arr = np.array(x_train[i])
        Final_train[int(c_train[i])-1].append(arr)
        #Final_train[c_train[i]-1].append(x_train[i])
    ans = []
    for i in range(nClasses):
        arr = np.array(Final_train[i])
        ans.append(arr)
    return np.array(ans)

def CrossCheck(train_data, train_data_class, dev_data):
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_data, train_data_class)
    train_data_reduced = clf.transform(train_data)
    dev_data_reduced = clf.transform(dev_data)
    return train_data_reduced, dev_data_reduced

def LDA(data, T, K):
    data = np.array(data, dtype = np.float64)
    nClasses = len(set(T))
    T_len = float(len(T))
    d = len(data[0])
    x_new, y_new = np.unique(T, return_inverse=True)  
    priors = np.bincount(y_new) / T_len

    classes = np.unique(T)
    SW = np.zeros((d,d))
    for idx, group in enumerate(classes):
        check = (T == group)
        data_reduced = data[check, :]
        cov = ComputeCovariance(data_reduced)
        SW += priors[idx] * np.atleast_2d(cov)
    
    ST = ComputeCovariance(data)
    SB = ST - SW

    if not np.all(np.linalg.eigvals(SW) > 0):
        temp = np.linalg.pinv(SW)
        eigValues, eigVectors = np.linalg.eigh(temp @ SB)
    else:
        eigValues,eigVectors = scipy.linalg.eigh(SB,SW)
    
    eigenVectors, eigenValues = SortMatrix(eigVectors, eigValues)
    eigenVectors_reduced = np.array(eigenVectors[:,:K])
    
    data = np.array(data)
    data_reduced = data @ eigenVectors_reduced
    return data_reduced, eigenVectors_reduced

def ReduceDimensions_LDA(train_data,train_data_class, dev_data, ndims):
    train_data_reduced, eigenVectors_reduced = LDA(train_data,train_data_class,ndims)
    dev_data_reduced = ProjectData(dev_data, eigenVectors_reduced)
    train_data_reduced, dev_data_reduced = ConvertToReal(train_data_reduced,dev_data_reduced)
    train_data_reduced, dev_data_reduced = CrossCheck(train_data, train_data_class, dev_data)
    return train_data_reduced, dev_data_reduced


# ## Image Dataset

# In[6]:


datasets_assigned = ['coast', 'forest', 'highway', 'mountain', 'opencountry']
def ReadInput_ImageDataset(subdir):
    data = []
    classes = []
    for dataset in range(len(datasets_assigned)):
        path = 'Features\\' + datasets_assigned[dataset] + '\\' + subdir
        for filename in os.scandir(path):
            with open(filename.path) as f:
                file_data = []
                for line in f:
                    values = list(map(float, line.strip().split(" ")))
                    for value in values:
                        file_data.append(value)
                data.append(file_data)
                classes.append(dataset)
    return data, classes


# In[7]:


train_data, train_data_class = ReadInput_ImageDataset('train')
dev_data, dev_data_class = ReadInput_ImageDataset('dev')
train_data, dev_data = Normalize(train_data, dev_data)
train_data_original, dev_data_original = train_data, dev_data
likelihoods_image = []
predictions_image = []


# In[8]:


K_range_Image = [20, 30]
numClasses_Image = 5 
accuracies, predictions, likelihoods = KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K_range_Image, numClasses_Image) 
likelihoods_image.append(likelihoods[0])
predictions_image.append(predictions[0])
print(accuracies)


# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, dev_data_class, numClasses_Image, K_range_Image, "ROC Curve for Image Dataset (kNN)", ["k = 20", "k = 30"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Image Dataset (kNN)", ["k = 20", "k = 30"])
# ConfusionMatrix(predictions[0], dev_data_class, numClasses_Image, "Image Dataset using kNN (k = 20)")

# ## Image Dataset with PCA

# In[10]:


train_data, dev_data = ReduceDimensions_PCA(train_data, dev_data, 80)
train_data_reduced, dev_data_reduced = train_data, dev_data
accuracies, predictions, likelihoods = KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K_range_Image, numClasses_Image) 
likelihoods_image.append(likelihoods[0])
predictions_image.append(predictions[0])
print(accuracies)


# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, dev_data_class, numClasses_Image, K_range_Image, "ROC Curve for Image Dataset (kNN)", ["k = 20", "k = 30"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Image Dataset (kNN)", ["k = 20", "k = 30"])
# ConfusionMatrix(predictions[0], dev_data_class, numClasses_Image, "Image Dataset using kNN (k = 20)")

# ## Image Dataset with LDA

# In[12]:


#K_range_Image = [4, 5]
train_data, dev_data = train_data_original, dev_data_original
train_data, dev_data = ReduceDimensions_LDA(train_data, train_data_class, dev_data, 4)
accuracies, predictions, likelihoods = KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K_range_Image, numClasses_Image) 
likelihoods_image.append(likelihoods[0])
predictions_image.append(predictions[0])
print(accuracies)


# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, dev_data_class, numClasses_Image, K_range_Image, "ROC Curve for Image Dataset (kNN)", ["k = 20", "k = 30"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Image Dataset (kNN)", ["k = 20", "k = 30"])
# ConfusionMatrix(predictions[0], dev_data_class, numClasses_Image, "Image Dataset using kNN (k = 20)")

# ## Image Dataset with PCA followed by LDA

# In[14]:


train_data, dev_data = train_data_reduced, dev_data_reduced
train_data, dev_data = ReduceDimensions_LDA(train_data, train_data_class, dev_data, 4)
accuracies, predictions, likelihoods = KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K_range_Image, numClasses_Image) 
likelihoods_image.append(likelihoods[0])
predictions_image.append(predictions[0])
print(accuracies)


# In[15]:


TPR_k, FPR_k, FNR_k = PlotROC(likelihoods_image, dev_data_class, 5, [1, 1, 1, 1], "ROC Curve for Image Dataset (KNN)", ["Original Dataset", "Reduced dataset (PCA)", "LDA on Original Dataset", "LDA on PCA-Reduced Dataset"])
PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Image Dataset (KNN)", ["Original Dataset", "Reduced dataset (PCA)", "LDA on Original Dataset", "LDA on PCA-Reduced Dataset"])
ConfusionMatrix(predictions_image[0], dev_data_class, 5, "Image Dataset using KNN")


# ## Synthetic Dataset 

# In[16]:


def ReadInput_Synthetic(file):
    x1_input, x2_input, z_input = [], [], []
    with open(file) as f:
        for line in f:
            lst = list(map(float, line.split(",")))
            x1_input.append(lst[0])
            x2_input.append(lst[1])
            z_input.append(lst[2])
    x1_input = np.array(x1_input)
    x2_input = np.array(x2_input)
    z_input  = np.array(z_input)
    return x1_input, x2_input, z_input


# In[17]:


x1_train, x2_train, y_train = ReadInput_Synthetic('16/train.txt')
x1_dev, x2_dev, y_dev = ReadInput_Synthetic('16/dev.txt')

for i in range(len(y_train)):
    y_train[i] -= 1
for i in range(len(y_dev)):
    y_dev[i] -= 1

synthetic_data_train = [[x1_train[i], x2_train[i]] for i in range(len(x1_train))]
synthetic_data_dev = [[x1_dev[i], x2_dev[i]] for i in range(len(x1_dev))]
synthetic_data_train_original, synthetic_data_dev_original = synthetic_data_train, synthetic_data_dev

likelihoods_synthetic = []
predictions_synthetic = []


# In[18]:


#100 percent accuracy for k = any of [5, 10, 15]
K_range = [5, 10, 15]
numClasses_Synthetic = 2
accuracies, predictions, likelihoods = KNN_Classifier(synthetic_data_train, y_train, synthetic_data_dev, y_dev, K_range, numClasses=numClasses_Synthetic)
likelihoods_synthetic.append(likelihoods[0])
predictions_synthetic.append(predictions[0])
print(accuracies)


# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, y_dev, numClasses_Synthetic, K_range, "ROC Curve for Synthetic Dataset (kNN)", ["k = 1", "k = 5", "k = 10"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Synthetic Data (kNN)")
# ConfusionMatrix(predictions[2], y_dev, numClasses_Synthetic, "Synthetic Dataset for k = 10 (kNN)")

# ## Synthetic Dataset with PCA

# In[20]:


# Reduce Synthetic Dataset to one dimension using PCA
synthetic_data_train, synthetic_data_dev = ReduceDimensions_PCA(synthetic_data_train, synthetic_data_dev, 1)
synthetic_data_train_reduced, synthetic_data_dev_reduced = synthetic_data_train, synthetic_data_dev
accuracies, predictions, likelihoods = KNN_Classifier(synthetic_data_train, y_train, synthetic_data_dev, y_dev, K_range, numClasses=numClasses_Synthetic)
likelihoods_synthetic.append(likelihoods[2])
predictions_synthetic.append(predictions[2])
print(accuracies)


# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, y_dev, numClasses_Synthetic, K_range, "ROC Curve for Synthetic Dataset (kNN)", ["k = 1", "k = 5", "k = 10"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Synthetic Data (kNN)", ["k = 5", "k = 10", "k = 15"])
# ConfusionMatrix(predictions[1], y_dev, numClasses_Synthetic, "Synthetic Dataset for k = 10 (kNN)")

# ## Synthetic Dataset with LDA

# In[22]:


synthetic_data_train, synthetic_data_dev = synthetic_data_train_original, synthetic_data_dev_original
synthetic_data_train, synthetic_data_dev = ReduceDimensions_LDA(synthetic_data_train, y_train, synthetic_data_dev, 1)
accuracies, predictions, likelihoods = KNN_Classifier(synthetic_data_train, y_train, synthetic_data_dev, y_dev, K_range, numClasses=numClasses_Synthetic)
likelihoods_synthetic.append(likelihoods[2])
predictions_synthetic.append(predictions[2])
print(accuracies)


# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, y_dev, numClasses_Synthetic, K_range, "ROC Curve for Synthetic Dataset (kNN)", ["k = 5", "k = 10", "k = 15"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Synthetic Data (kNN)")
# ConfusionMatrix(predictions[2], y_dev, numClasses_Synthetic, "Synthetic Dataset for k = 10 (kNN)")

# ## Synthetic Data with PCA followed by LDA

# In[24]:


synthetic_data_train, synthetic_data_dev = synthetic_data_train_reduced, synthetic_data_dev_reduced
synthetic_data_train, synthetic_data_dev = ReduceDimensions_LDA(synthetic_data_train, y_train, synthetic_data_dev, 1)
accuracies, predictions, likelihoods = KNN_Classifier(synthetic_data_train, y_train, synthetic_data_dev, y_dev, K_range, numClasses=numClasses_Synthetic)
likelihoods_synthetic.append(likelihoods[2])
predictions_synthetic.append(predictions[2])
print(accuracies)


# In[25]:


TPR_k, FPR_k, FNR_k = PlotROC(likelihoods_synthetic, y_dev, 2, [1, 1, 1], "ROC Curve for Synthetic Dataset (KNN)", ["Original Dataset", "Reduced Dataset (PCA)", "LDA on Original Dataset"])
PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Synthetic Dataset (KNN)", ["Original Dataset", "Reduced Dataset (PCA)", "LDA on Original Dataset"])
ConfusionMatrix(predictions_synthetic[0], y_dev, 2, "Synthetic Dataset using KNN")


# ## Handwriting Dataset 

# In[26]:


datasets = ['a', 'ai', 'chA', 'lA', 'tA']    # datasets assigned to team 16
mapping = {'a' : 0 ,  'ai' : 1 , 'chA' : 2, 'lA' : 3, 'tA' : 4}

def ReadInput_Handwriting(subdir):
    data = []
    data_class = []
    for letter in datasets:
        directory = 'Handwriting_Data\\' + letter + '\\' + subdir
        for filename in os.scandir(directory):
            with open(filename.path) as f:
                for line in f:
                    values = line.strip().split(" ")
                    NC = values[0]
                    l = len(values)
                    x = [float(values[i]) for i in range(1, l, 2)]
                    y = [float(values[i+1]) for i in range(1, l, 2)]
                    max_x, min_x = max(x), min(x)
                    max_y, min_y = max(y), min(y)
                    points = [((x[i] - min_x)/(max_x - min_x), (y[i] - min_y)/(max_y - min_y)) for i in range(len(x))]
                    data.append(points)
                    data_class.append(mapping[letter])
    return data, data_class


# In[27]:


# Functions to extract constant length features from data
def Reduce(data_point, target_len):
    n = len(data_point)
    window = n - target_len + 1
    data_point = np.array(data_point)
    smooth_data_point = []
    #print(data_point)
    for i in range(target_len):
        current_arr = data_point[i:i+window]
        avg = np.mean(current_arr, axis=0)
        smooth_data_point.append(avg)
    smooth_data_point = np.array(smooth_data_point)
    smooth_data_point = smooth_data_point.flatten()
    return smooth_data_point

def SmoothenData(train_data, dev_data):
    smooth_train_data = []
    smooth_dev_data = []
        
    min_length = len(train_data[0])
    for i in range(len(train_data)):
        if(len(train_data[i]) < min_length):
            min_length = len(train_data[i])
    for i in range(len(dev_data)):
        if(len(dev_data[i]) < min_length):
            min_length = len(dev_data[i])
    
    for i in range(len(train_data)):
        smooth_train_point = Reduce(train_data[i], min_length)
        smooth_train_data.append(smooth_train_point)
    for i in range(len(dev_data)):
        smooth_dev_point = Reduce(dev_data[i], min_length)
        smooth_dev_data.append(smooth_dev_point)
    
    return smooth_train_data, smooth_dev_data


# In[28]:


train_data, train_data_class = ReadInput_Handwriting('train')
dev_data, dev_data_class = ReadInput_Handwriting('dev')
train_data, dev_data = SmoothenData(train_data, dev_data)
train_data_original, dev_data_original = train_data, dev_data
likelihoods_handwriting = []
predictions_handwriting = []


# In[29]:


K_range_Handwriting = [5, 10, 15]
numClasses_Handwriting = 5
accuracies, predictions, likelihoods = KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K_range_Handwriting, numClasses_Handwriting) 
likelihoods_handwriting.append(likelihoods[2])
predictions_handwriting.append(predictions[2])
print(accuracies)


# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, dev_data_class, numClasses_Handwriting, K_range_Handwriting, "ROC Curve for Handwriting Dataset (kNN)", ["k = 5", "k = 10", "k = 15"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Handwriting Dataset (kNN)", ["k = 5", "k = 10", "k = 15"])
# ConfusionMatrix(predictions[0], dev_data_class, numClasses_Handwriting, "Handwriting Dataset using kNN (k = 5)")

# ## Handwritten Data with PCA

# In[31]:


train_data, dev_data = ReduceDimensions_PCA(train_data, dev_data, 150)
train_data_reduced, dev_data_reduced = train_data, dev_data
accuracies, predictions, likelihoods = KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K_range_Handwriting, numClasses_Handwriting) 
likelihoods_handwriting.append(likelihoods[2])
predictions_handwriting.append(predictions[2])
print(accuracies)


# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, dev_data_class, numClasses_Handwriting, K_range_Handwriting, "ROC Curve for Handwriting Dataset (kNN)", ["k = 5", "k = 10", "k = 15"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Handwriting Dataset (kNN)", ["k = 5", "k = 10", "k = 15"])
# ConfusionMatrix(predictions[0], dev_data_class, numClasses_Handwriting, "Handwriting Dataset using kNN (k = 5)")

# ## Handwriting Dataset with LDA

# In[33]:


train_data, dev_data = train_data_original, dev_data_original
train_data, dev_data = ReduceDimensions_LDA(train_data, train_data_class, dev_data, 4)
accuracies, predictions, likelihoods = KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K_range_Handwriting, numClasses_Handwriting) 
likelihoods_handwriting.append(likelihoods[2])
predictions_handwriting.append(predictions[2])
print(accuracies)


# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, dev_data_class, numClasses_Handwriting, K_range_Handwriting, "ROC Curve for Handwriting Dataset (kNN)", ["k = 5", "k = 10", "k = 15"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Handwriting Dataset (kNN)", ["k = 1", "k = 5", "k = 10"])
# ConfusionMatrix(predictions[0], dev_data_class, numClasses_Handwriting, "Handwriting Dataset using kNN (k = 1)")

# ## Handwriting Dataset with PCA followed by LDA

# In[35]:


train_data, dev_data = train_data_reduced, dev_data_reduced
train_data, dev_data = ReduceDimensions_LDA(train_data, train_data_class, dev_data, 4)
accuracies, predictions, likelihoods = KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K_range_Handwriting, numClasses_Handwriting) 
likelihoods_handwriting.append(likelihoods[2])
predictions_handwriting.append(predictions[2])
print(accuracies)


# In[36]:


TPR_k, FPR_k, FNR_k = PlotROC(likelihoods_handwriting, dev_data_class, 5, [1, 1, 1, 1], "ROC Curve for Handwriting Dataset (KNN)", ["Original Dataset", "Reduced dataset (PCA)", "LDA on Original Dataset", "LDA on PCA-Reduced Dataset"])
PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Handwriting Dataset (KNN)", ["Original Dataset", "Reduced dataset (PCA)", "LDA on Original Dataset", "LDA on PCA-Reduced Dataset"])
ConfusionMatrix(predictions_handwriting[0], dev_data_class, 5, "Handwriting Dataset using KNN")


# ## Spoken Digit Dataset 

# In[37]:


datasets = ['4', '5', '7', '8', 'o']    # datasets assigned to team 16
mapping = {'4' : 0 , '5' : 1 , '7' : 2, '8' : 3, 'o' : 4}

def ReadInput_SpokenDigit(subdir):
    data = []
    data_class = []
    for letter in datasets:
        directory = 'Isolated_Digits\\' + letter + '\\' + subdir
        for filename in os.scandir(directory):
            file_data = []
            if filename.path[-4:] == 'mfcc':
                with open(filename.path) as f:
                    line = f.readline()
                    values = line.strip().split(" ")
                    for line in f.readlines():
                        values = list(map(float, line.strip().split(" ")))
                        file_data.append(values) 
                data.append(file_data)
                data_class.append(mapping[letter])
    return data, data_class


# In[38]:


train_data, train_data_class = ReadInput_SpokenDigit('train')
dev_data, dev_data_class = ReadInput_SpokenDigit('dev')
train_data, dev_data = SmoothenData(train_data, dev_data)
train_data_original, dev_data_original = train_data, dev_data
likelihoods_spoken = []
predictions_spoken = []


# In[39]:


K_range_Spoken = [5, 10, 15]
numClasses_Spoken = 5
accuracies, predictions, likelihoods = KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K_range_Spoken, numClasses_Spoken) 
likelihoods_spoken.append(likelihoods[1])
predictions_spoken.append(predictions[1])
print(accuracies)


# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, dev_data_class, numClasses_Spoken, K_range_Handwriting, "ROC Curve for Spoken Digit Dataset (kNN)", ["k = 1", "k = 5", "k = 10"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Spoken Digit Dataset (kNN)", ["k = 5", "k = 10", "k = 10"])
# ConfusionMatrix(predictions[0], dev_data_class, numClasses_Spoken, "Spoken-Digit Dataset using kNN (k = 10)")

# ## Spoken Digit Dataset with PCA 

# In[41]:


train_data, dev_data = ReduceDimensions_PCA(train_data, dev_data, 100)
train_data_reduced, dev_data_reduced = train_data, dev_data
accuracies, predictions, likelihoods = KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K_range_Spoken, numClasses_Spoken) 
likelihoods_spoken.append(likelihoods[1])
predictions_spoken.append(predictions[1])
print(accuracies)


# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, dev_data_class, numClasses_Spoken, K_range_Handwriting, "ROC Curve for Spoken Digit Dataset (kNN)", ["k = 1", "k = 5", "k = 10"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Spoken Digit Dataset (kNN)", ["k = 1", "k = 10", "k = 20"])
# ConfusionMatrix(predictions[0], dev_data_class, numClasses_Spoken, "Spoken-Digit Dataset using kNN (k = 1)")

# ## Spoken Digit Dataset with LDA

# In[43]:


train_data, dev_data = train_data_original, dev_data_original
train_data, dev_data = ReduceDimensions_LDA(train_data, train_data_class, dev_data, 4)
accuracies, predictions, likelihoods = KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K_range_Spoken, numClasses_Spoken) 
likelihoods_spoken.append(likelihoods[2])
predictions_spoken.append(predictions[2])
print(accuracies)


# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, dev_data_class, numClasses_Spoken, K_range_Handwriting, "ROC Curve for Spoken Digit Dataset (kNN)", ["k = 1", "k = 5", "k = 10"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Spoken Digit Dataset (kNN)", ["k = 1", "k = 10", "k = 20"])
# ConfusionMatrix(predictions[0], dev_data_class, numClasses_Spoken, "Spoken-Digit Dataset using kNN (k = 1)")

# ## Spoken-Digit Dataset with PCA followed by LDA

# In[45]:


train_data, dev_data = train_data_reduced, dev_data_reduced
train_data, dev_data = ReduceDimensions_LDA(train_data, train_data_class, dev_data, 4)
accuracies, predictions, likelihoods = KNN_Classifier(train_data, train_data_class, dev_data, dev_data_class, K_range_Spoken, numClasses_Spoken) 
likelihoods_spoken.append(likelihoods[2])
predictions_spoken.append(predictions[2])
print(accuracies)


# In[46]:


TPR_k, FPR_k, FNR_k = PlotROC(likelihoods_spoken, dev_data_class, 5, [1, 1, 1, 1], "ROC Curve for Spoken-Digit Dataset (KNN)", ["Original Dataset", "Reduced dataset (PCA)", "LDA on Original Dataset", "LDA on PCA-Reduced Dataset"])
PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Spoken-Digit Dataset (KNN)", ["Original Dataset", "Reduced dataset (PCA)", "LDA on Original Dataset", "LDA on PCA-Reduced Dataset"])
ConfusionMatrix(predictions_spoken[0], dev_data_class, 5, "Spoken-Digit Dataset using KNN")


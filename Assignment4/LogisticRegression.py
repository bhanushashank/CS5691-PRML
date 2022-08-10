#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import DetCurveDisplay
import seaborn as sns
from scipy.stats import multivariate_normal 
import scipy
from sklearn.preprocessing import MinMaxScaler


# # Helper functions for logistic Regression

# In[2]:


def Normalize(train_data,dev_data):
    scaling = MinMaxScaler().fit(train_data)
    train_scaled = scaling.transform(train_data)
    dev_scaled = scaling.transform(dev_data)   
    return train_scaled,dev_scaled


# # Logistic Regression

# In[3]:


def parameterEstimation(x,weights):
    length = len(weights)
    ans = 0
    for i in range(1,length):
        ans += x[i-1]*weights[i]
    ans += weights[0]
    return ans
    
def likelihood(x,a,classNumber):
    Total = 0
    for i in range(len(a)):
        Total += math.exp(parameterEstimation(x,a[i]))
    return math.exp(parameterEstimation(x,a[classNumber]))/Total

def LogisticRegression(x_train,c_train,nClasses,it,lf = 10**(-5)):
    d = len(x_train[0])
    w_old = np.zeros((nClasses,d+1))
    Iterations = it
    learningFactor = lf
    for i in range(Iterations):
        w_new = np.zeros((nClasses,d+1))
        for j in range(nClasses):
            ans = np.zeros(d+1)
            for k in range(len(x_train)):
                temp = likelihood(x_train[k],w_old,j)
                if(c_train[k]==j):
                    temp -= 1
                arr = np.array(x_train[k])
                arr = np.insert(arr,0,1)
                ans = ans + temp * arr
            w_new[j] = np.subtract(w_old[j],learningFactor*ans)
        w_old = w_new
    return w_old

def prediction(x_train,c_train,x_dev,groundTruth,iterations,lf = 10**(-5)):
    correctPredict = 0
    w = LogisticRegression(x_train,c_train,len(set(c_train)),iterations,lf)
    probs = []
    prediction_class = []
    for i in range(len(x_dev)):
        temp = []
        probs.append([])
        for j in range(len(set(c_train))):
            ans = likelihood(x_dev[i],w,j)
            probs[i].append(ans)
            temp.append((ans,j))
        temp.sort(reverse = True)
        prediction_class.append(temp[0][1])
        if(groundTruth[i] == temp[0][1]):
            correctPredict += 1
    print(correctPredict/len(x_dev))
    return probs,prediction_class


# def parameterEstimation2(x,weights):
#     ans = weights[0]
#     d = len(x)
#     count = 1
#     '''
#     for i in range(d):
#         ans += weights[count]*x[i]
#         count += 1
#     '''
#     for i in range(d):
#         for j in range(i,d):
#             ans += weights[count]*x[i]*x[j]
#             count += 1
#     return ans
#     
# def likelihood2(x,a,classNumber):
#     Total = 0
#     for i in range(len(a)):
#         Total += math.exp(parameterEstimation2(x,a[i]))
#     return math.exp(parameterEstimation2(x,a[classNumber]))/Total
# 
# 
# def polynomial2(x):
#     d = len(x)
#     ans = [1]
#     '''
#     for i in range(d):
#         ans.append(x[i])
#     '''
#     for i in range(d):
#         for j in range(i,d):
#             ans.append(x[i]*x[j])
#     return ans
# 
# def LogisticRegression(x_train,c_train,nClasses,it,lf = 10**(-5)):
#     d = len(x_train[0])
#     Iterations = it
#     learningFactor = lf
#     #basislength = 1 + d + ((d*(d+1))//2)
#     basislength = 1 + ((d*(d+1))//2)
#     w_old = np.zeros((nClasses,basislength))
#     for i in range(Iterations):
#         w_new = np.zeros((nClasses,basislength))
#         for j in range(nClasses):
#             ans = np.zeros(basislength)
#             for k in range(len(x_train)):
#                 temp = likelihood2(x_train[k],w_old,j)
#                 if(c_train[k]==j):
#                     temp -= 1
#                 arr = np.array(polynomial2(x_train[k]))
#                 ans = ans + temp * arr
#             w_new[j] = np.subtract(w_old[j],learningFactor*ans)
#         w_old = w_new
#     return w_old
# 
# def prediction(x_train,c_train,x_dev,groundTruth,iterations,lf = 10**(-5)):
#     correctPredict = 0
#     w = LogisticRegression(x_train,c_train,len(set(c_train)),iterations,lf)
#     probs = []
#     prediction_class = []
#     for i in range(len(x_dev)):
#         temp = []
#         probs.append([])
#         for j in range(len(set(c_train))):
#             ans = likelihood2(x_dev[i],w,j)
#             probs[i].append(ans)
#             temp.append((ans,j))
#         temp.sort(reverse = True)
#         prediction_class.append(temp[0][1])
#         if(groundTruth[i] == temp[0][1]):
#             correctPredict += 1
#     print(correctPredict/len(x_dev))
#     return probs,prediction_class

# ## Functions for ROC, DET, Confusion Matrix

# In[4]:


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

# In[5]:


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

# In[ ]:


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


# ## Synthetic Dataset without PCA or LDA

# In[7]:


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


# In[8]:


x1_train, x2_train, y_train = ReadInput_Synthetic('16/train.txt')
x1_dev, x2_dev, y_dev = ReadInput_Synthetic('16/dev.txt')

for i in range(len(y_train)):
    y_train[i] -= 1
for i in range(len(y_dev)):
    y_dev[i] -= 1

synthetic_data_train_original = [[x1_train[i], x2_train[i]] for i in range(len(x1_train))]
synthetic_data_dev_original = [[x1_dev[i], x2_dev[i]] for i in range(len(x1_dev))]
synthetic_likelihoods = []
synthetic_data_train_original,synthetic_data_dev_original = Normalize(synthetic_data_train_original,synthetic_data_dev_original)


# In[9]:


#100 percent accuracy for k = any of [1, 5, 10]
numClasses_Synthetic = 2
synthetic_data_train = synthetic_data_train_original
synthetic_data_dev = synthetic_data_dev_original
likelihoods,predictions = prediction(synthetic_data_train,y_train,synthetic_data_dev,y_dev,100)
synthetic_likelihoods.append(likelihoods)


# ## Synthetic Dataset with PCA

# In[10]:


# Reduce Synthetic Dataset to one dimension using PCA
synthetic_data_train = synthetic_data_train_original
synthetic_data_dev = synthetic_data_dev_original
synthetic_data_train, synthetic_data_dev = ReduceDimensions_PCA(synthetic_data_train, synthetic_data_dev, 1)


# In[11]:


likelihoods,predictions = prediction(synthetic_data_train,y_train,synthetic_data_dev,y_dev,500)
synthetic_likelihoods.append(likelihoods)


# likelihoods = [likelihoods]
# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, y_dev, numClasses_Synthetic,[1], "ROC Curve for Synthetic Dataset (Logistic Regression)", ["k = 1", "k = 5", "k = 10"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Synthetic Data (Logistic Regression)")
# ConfusionMatrix(predictions, y_dev, numClasses_Synthetic, "Synthetic Dataset (Logistic Regression)")

# ## Synthetic Dataset with LDA

# In[12]:


synthetic_data_train = synthetic_data_train_original
synthetic_data_dev = synthetic_data_dev_original
synthetic_data_train, synthetic_data_dev = ReduceDimensions_LDA(synthetic_data_train, y_train, synthetic_data_dev, 1)


# In[13]:


likelihoods,predictions = prediction(synthetic_data_train,y_train,synthetic_data_dev,y_dev,500)
synthetic_likelihoods.append(likelihoods)


# # Plotting the ROC and DET curve

# In[14]:


TPR_k, FPR_k, FNR_k = PlotROC(synthetic_likelihoods, y_dev, 2, [1,2,3] , "ROC Curve for Synthetic Dataset (Logistic Regression)", ["without PCA and LDA","with PCA","with LDA"])
PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Synthetic Dataset (Logistic Regression)", ["without PCA and LDA","with PCA","with LDA"])
ConfusionMatrix(predictions, y_dev, numClasses_Synthetic, "Synthetic Dataset using Logistic Regression")


# ## Image Dataset

# In[7]:


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


# In[8]:


train_data, train_data_class = ReadInput_ImageDataset('train')
dev_data, dev_data_class = ReadInput_ImageDataset('dev')
#train_data = Normalize(train_data)
#dev_data = Normalize(dev_data)
train_data , dev_data = Normalize(train_data,dev_data)
Image_likelihoods = []


# In[17]:


K_range_Image = [20, 30]
numClasses_Image = 5 
likelihoods , predictions = prediction(train_data,train_data_class,dev_data,dev_data_class,50)
Image_likelihoods.append(likelihoods)


# ## Image Dataset with PCA

# In[18]:


train_data_pca,dev_data_pca = ReduceDimensions_PCA(train_data,dev_data,60)


# In[19]:


likelihoods,predictions = prediction(train_data_pca,train_data_class,dev_data_pca,dev_data_class,500)
Image_likelihoods.append(likelihoods)
ConfusionMatrix(predictions, dev_data_class, 5 , "Image Dataset using Logistic Regression")


# # Image Dataset with PCA followed by LDA

# In[20]:


train_data_lda,dev_data_lda = ReduceDimensions_LDA(train_data_pca,train_data_class,dev_data_pca,4)


# In[21]:


likelihoods,predictions = prediction(train_data_lda,train_data_class,dev_data_lda,dev_data_class,350,10**(-2.5))
Image_likelihoods.append(likelihoods)


# ## Image Dataset with LDA

# In[11]:


train_data_lda,dev_data_lda = ReduceDimensions_LDA(train_data,train_data_class,dev_data,4)
likelihoods,predictions = prediction(train_data_lda,train_data_class,dev_data_lda,dev_data_class,1000,10**(-2.5))
Image_likelihoods.append(likelihoods)


# # ROC and DET for Image Dataset

# In[23]:


TPR_k, FPR_k, FNR_k = PlotROC(Image_likelihoods, dev_data_class, numClasses_Image, [1,2,3,4] , "ROC Curve for Image Dataset (Logistic Regression)", ["without PCA and LDA","with PCA","with PCA followed by LDA","with LDA"])
PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Image Dataset (Logistic Regression)", ["without PCA and LDA","with PCA","with PCA followed by LDA","with LDA"])
ConfusionMatrix(predictions, dev_data_class, numClasses_Image, "Image Dataset using Logistic Regression")


# ## Handwriting Dataset without PCA or LDA

# In[24]:


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
                    points = [[(x[i] - min_x)/(max_x - min_x), (y[i] - min_y)/(max_y - min_y)] for i in range(len(x))]
                    data.append(points)
                    data_class.append(mapping[letter])
    return data, data_class


# In[25]:


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


# In[26]:


train_data, train_data_class = ReadInput_Handwriting('train')
dev_data, dev_data_class = ReadInput_Handwriting('dev')
train_data, dev_data = SmoothenData(train_data, dev_data)
train_data , dev_data = Normalize(train_data,dev_data)
Handwritten_likelihoods = []


# In[27]:


K_range_Handwriting = [1, 5, 10]
numClasses_Handwriting = 5
likelihoods,predictions = prediction(train_data,train_data_class,dev_data,dev_data_class,1500,10**(-2.5))
Handwritten_likelihoods.append(likelihoods)
ConfusionMatrix(predictions, dev_data_class, 5 , "Handwriting Dataset using Logistic Regression")


# ## Handwritten Data with PCA

# In[28]:


train_data_pca,dev_data_pca = ReduceDimensions_PCA(train_data,dev_data,10)
likelihoods,predictions = prediction(train_data_pca,train_data_class,dev_data_pca,dev_data_class,500,10**(-2.5))
Handwritten_likelihoods.append(likelihoods)


# likelihoods = [likelihoods]
# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, dev_data_class, numClasses_Handwriting,[1], "ROC Curve for Handwriting Dataset (Logistic Regression)", ["k = 1", "k = 5", "k = 10"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Handwriting Dataset (Logistic Regression)", ["k = 1", "k = 5", "k = 10"])
# ConfusionMatrix(predictions, dev_data_class, numClasses_Handwriting, "Handwriting Dataset using Logistic Regression (k = 1)")

# # Handwritten Data with PCA followed by LDA

# In[29]:


train_data_lda,dev_data_lda = ReduceDimensions_LDA(train_data_pca,train_data_class,dev_data_pca,4)


# In[30]:


likelihoods,predictions = prediction(train_data_lda,train_data_class,dev_data_lda,dev_data_class,500,10**(-2.5))
Handwritten_likelihoods.append(likelihoods)


# ## Handwriting Dataset with LDA

# In[31]:


train_data_lda,dev_data_lda = ReduceDimensions_LDA(train_data,train_data_class,dev_data,4)
likelihoods,predictions = prediction(train_data_lda,train_data_class,dev_data_lda,dev_data_class,500,10**(-2.5))
Handwritten_likelihoods.append(likelihoods)


# likelihoods = [likelihoods]
# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, dev_data_class, numClasses_Handwriting, [1], "ROC Curve for Handwriting Dataset (Logistic Regression)", ["k = 1", "k = 5", "k = 10"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Handwriting Dataset (Logistic Regression)", ["k = 1", "k = 5", "k = 10"])
# ConfusionMatrix(predictions, dev_data_class, numClasses_Handwriting, "Handwriting Dataset using Logistic Regression (k = 1)")

# # Plot ROC and DET 

# In[32]:


TPR_k, FPR_k, FNR_k = PlotROC(Handwritten_likelihoods, dev_data_class, 5 , [1,2,3,4] , "ROC Curve for Handwriting Dataset (Logistic Regression)", ["without PCA and LDA","with PCA","with PCA followed by LDA","with LDA"])
PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Handwriting Dataset (Logistic Regression)", ["without PCA and LDA","with PCA","with PCA followed by LDA","with LDA"])
ConfusionMatrix(predictions, dev_data_class, 5 , "Handwriting Dataset using Logistic Regression")


# ## Spoken Digit Dataset without PCA or LDA

# In[33]:


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


# In[34]:


train_data, train_data_class = ReadInput_SpokenDigit('train')
dev_data, dev_data_class = ReadInput_SpokenDigit('dev')
train_data, dev_data = SmoothenData(train_data, dev_data)
Spoken_likelihoods = []
#train_data = Normalize(train_data)
#dev_data = Normalize(dev_data)
#train_data ,dev_data = Normalize(train_data,dev_data)


# In[35]:


K_range_Spoken = [1, 5, 10]
numClasses_Spoken = 5


# In[36]:


likelihoods,predictions = prediction(train_data,train_data_class,dev_data,dev_data_class,50)
Spoken_likelihoods.append(likelihoods)


# ## Spoken Digit Dataset with PCA 

# In[37]:


train_data_pca,dev_data_pca = ReduceDimensions_PCA(train_data,dev_data,50)


# In[38]:


likelihoods,predictions = prediction(train_data_pca,train_data_class,dev_data_pca,dev_data_class,500)
Spoken_likelihoods.append(likelihoods)


# # Spoken data for PCA followed by LDA

# In[39]:


train_data_lda,dev_data_lda = ReduceDimensions_LDA(train_data_pca,train_data_class,dev_data_pca,4)


# In[40]:


likelihoods,predictions = prediction(train_data_lda,train_data_class,dev_data_lda,dev_data_class,500)
Spoken_likelihoods.append(likelihoods)
ConfusionMatrix(predictions, dev_data_class, 5 , "Spoken Dataset using Logistic Regression")


# ## Spoken Digit Dataset with LDA

# In[41]:


train_data_lda,dev_data_lda = ReduceDimensions_LDA(train_data,train_data_class,dev_data,4)


# In[42]:


likelihoods,predictions = prediction(train_data_lda,train_data_class,dev_data_lda,dev_data_class,500)
Spoken_likelihoods.append(likelihoods)


# likelihoods = [likelihoods]
# TPR_k, FPR_k, FNR_k = PlotROC(likelihoods, dev_data_class, numClasses_Spoken, [1], "ROC Curve for Spoken Digit Dataset (Logistic Regression)", ["k = 1", "k = 5", "k = 10"])
# PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Spoken Digit Dataset (Logistic Regression)", ["k = 1", "k = 10", "k = 20"])
# ConfusionMatrix(predictions, dev_data_class, numClasses_Spoken, "Spoken-Digit Dataset using Logistic Regression (k = 1)")

# # plot ROC and DET for spoken digit

# In[43]:


TPR_k, FPR_k, FNR_k = PlotROC(Spoken_likelihoods, dev_data_class, 5 , [1,2,3,4] , "ROC Curve for Spoken Dataset (Logistic Regression)", ["without PCA and LDA","with PCA","with PCA followed by LDA","with LDA"])
PlotDET(TPR_k, FPR_k, FNR_k, "DET Curve for Spoken Dataset (Logistic Regression)", ["without PCA and LDA","with PCA","with PCA followed by LDA","with LDA"])
ConfusionMatrix(predictions, dev_data_class, 5 , "Spoken Dataset using Logistic Regression")


# In[ ]:





# In[ ]:





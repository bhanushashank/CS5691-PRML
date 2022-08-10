#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import DetCurveDisplay
import seaborn as sns
from scipy.stats import multivariate_normal 


# In[2]:


def ReadInput(file):
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

def MultivariateGaussian(x, mean, cov):
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    constant = 1.0 / ((2 * math.pi) ** (len(x) / 2) * (cov_det ** 0.5)) 
    diff = np.array([x[i] - mean[i] for i in range(len(x))])
    exp = math.exp((-1.0 / 2) * ((diff @ cov_inv) @ diff.T))
    return constant * exp


# ## K-Means Clustering

# In[3]:


def NotEqual(x1, x2):
    if len(x1) != len(x2):
        return False
    for i in range(len(x1)):
        if x1[i] != x2[i]:
            return True
    return False

def EuclideanDistance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i])**2
    return math.sqrt(distance)

# Perform K-Means Clustering on the given data with K clusters.
def KMeans(data, K, iteration_limit=100):
    N = len(data)
    old_clusterInfo = np.zeros(N)    
    new_clusterInfo = np.zeros(N)
    freq = np.zeros(K)
    
    # Pick K random points from the dataset as initial cluster means
    random_indices = random.sample(range(0, N), K)
    means = [np.array(data[idx]) for idx in random_indices]
    
    old_clusterInfo[0] = 1
    iterations = 0
    
    # Repeat until convergence or iteration_limit times, whichever reaches first
    while(NotEqual(old_clusterInfo, new_clusterInfo) and iterations <= iteration_limit):
        old_clusterInfo = new_clusterInfo
        iterations += 1
        
        # Assign data points to their closest clusters (means)
        for i in range(N):
            shortest = EuclideanDistance(means[0], data[i])
            cluster = 0
            for j in range(K):
                distance = EuclideanDistance(list(means[j]), data[i])
                if(shortest > distance):
                    shortest = distance
                    cluster = j
            new_clusterInfo[i] = cluster   # Assign data[i] to its closest cluster
            freq[cluster] += 1             # Increase frequency of the closest cluster by 1
        
        # Compute new cluster means using the reassigned data points to the clusters
        means = [np.zeros(len(data[0])) for i in range(K)]
        for i in range(N):
            means[int(new_clusterInfo[i])] = means[int(new_clusterInfo[i])] + data[i]
        for i in range(K):
            means[i] = means[i] * (1/freq[i])
            means[i] = list(means[i])
        
    return means, new_clusterInfo, freq


# ## Density estimation using GMMs

# In[4]:


# Returns A.T @ A for a row vector A
def SelfMultiply(a):
    matrix = []
    for i in range(len(a)):
        matrix.append([])
    for i in range(len(a)):
        for j in range(len(a)):
            matrix[i].append(a[i]*a[j])
    return matrix

# Use K-Means algorithm to find initial parameters for GMMs
def InitialEstimate(data, K):
    N = len(data)
    new_covariance = []
    for i in range(K):
        new_covariance.append(np.zeros((len(data[0]), len(data[0]))))
        
    means, clusterInfo, priors = KMeans(data, K)
    clusterInfo = np.array(clusterInfo, dtype=int)
        
    for i in range(N):
        diff = np.subtract(np.array(data[i]), np.array(means[clusterInfo[i]]))
        cov = np.array(SelfMultiply(diff))
        new_covariance[clusterInfo[i]] = np.add(new_covariance[clusterInfo[i]], cov)
    for i in range(K):
        new_covariance[i] = new_covariance[i] * (1/priors[i])
        priors[i] /= N
    return means, new_covariance, priors

# Find the parameters of GMMs for each class using EM algorithm
def GMM(data, K, initialPriors, initialMeans, initialCovariances):
    N = len(data)
    gamma = np.zeros((N, K))
    
    priors = initialPriors
    means = initialMeans
    covariances = initialCovariances
    
    for count in range(4):
        new_priors = np.zeros(K)
        new_means = [[] for i in range(K)]
        new_covariances = []
        for i in range(K):
            new_covariances.append(np.zeros((len(data[0]), len(data[0]))))
            
        for i in range(N):
            total = 0
            for j in range(K):
                total += priors[j] * MultivariateGaussian(data[i], means[j], covariances[j])
            for j in range(K):
                gamma[i][j] = priors[j] * MultivariateGaussian(data[i], means[j], covariances[j])/total
        
        for i in range(K):
            mixture_mean = np.zeros(len(data[0]))
            mixture_covariance = []
            mixture_covariance.append(np.zeros((len(data[0]), len(data[0]))))
            
            N_k = 0
            for j in range(N):
                N_k += gamma[j][i]
                mixture_mean = np.add(mixture_mean, np.array(data[j]) * gamma[j][i])
                diff = np.subtract(np.array(data[j]), np.array(means[i]))
                cov = (np.array(SelfMultiply(diff))) * gamma[j][i]
                mixture_covariance = np.add(mixture_covariance, cov) 

            new_priors[i] = N_k/N
            new_means[i] = mixture_mean * (1/N_k)
            new_covariances[i] = mixture_covariance * (1/N_k)
        
        priors = new_priors
        means = new_means
        covariances = new_covariances
    return means, covariances, priors


# In[5]:


# Use K-Means algorithm to find initial parameters for GMMs
def InitialEstimate_Diagonal(data, K):
    N = len(data)
    new_covariance = []
    for i in range(K):
        new_covariance.append(np.zeros((len(data[0]), len(data[0]))))
        
    means, clusterInfo, priors = KMeans(data, K)
    clusterInfo = np.array(clusterInfo, dtype=int)
        
    for i in range(N):
        diff = np.subtract(np.array(data[i]), np.array(means[clusterInfo[i]]))
        cov = np.array(SelfMultiply(diff))
        new_covariance[clusterInfo[i]] = np.add(new_covariance[clusterInfo[i]], cov)
    for i in range(K):
        new_covariance[i] = new_covariance[i] * (1/priors[i])
        priors[i] /= N
    for i in range(K):
        for j in range(len(data[0])):
            for k in range(len(data[0])):
                if(j != k):
                    new_covariance[i][j][k] = 0
    return means, new_covariance, priors

# Find the parameters of GMMs for each class using EM algorithm
def GMM_Diagonal(data, K, initialPriors, initialMeans, initialCovariances):
    N = len(data)
    gamma = np.zeros((N, K))
    
    priors = initialPriors
    means = initialMeans
    covariances = initialCovariances
    
    for count in range(4):
        new_priors = np.zeros(K)
        new_means = [[] for i in range(K)]
        new_covariances = []
        for i in range(K):
            new_covariances.append(np.zeros((len(data[0]), len(data[0]))))
            
        for i in range(N):
            total = 0
            for j in range(K):
                total += priors[j] * MultivariateGaussian(data[i], means[j], covariances[j])
            for j in range(K):
                gamma[i][j] = priors[j] * MultivariateGaussian(data[i], means[j], covariances[j])/total
        
        for i in range(K):
            mixture_mean = np.zeros(len(data[0]))
            mixture_covariance = []
            mixture_covariance.append(np.zeros((len(data[0]), len(data[0]))))
            
            N_k = 0
            for j in range(N):
                N_k += gamma[j][i]
                mixture_mean = np.add(mixture_mean, np.array(data[j]) * gamma[j][i])
                diff = np.subtract(np.array(data[j]), np.array(means[i]))
                cov = (np.array(SelfMultiply(diff))) * gamma[j][i]
                mixture_covariance = np.add(mixture_covariance, cov) 

            new_priors[i] = N_k/N
            new_means[i] = mixture_mean * (1/N_k)
            new_covariances[i] = mixture_covariance * (1/N_k)
            
        for i in range(K):
            for j in range(len(data[0])):
                for k in range(len(data[0])):
                    if(j != k):
                        new_covariances[i][0][j][k] = 0
        priors = new_priors
        means = new_means
        covariances = new_covariances
    return means, covariances, priors


# In[6]:


def classificationPrediction(data,y,C,K,Finalmeans,Finalcovariances,Finalpriors):
    C_Prediction = 0
    N = len(data)
    for i in range(N):
        prob_density = Density(data[i],C,K,Finalmeans,Finalcovariances,Finalpriors)
        temp = []
        for j in range(C):
            temp.append((prob_density[j],j))
        temp.sort(reverse=True)
        if(temp[0][1]==(y[i]-1)):
            C_Prediction += 1
    return C_Prediction/N


# In[7]:


def Density(x, C, K, means, covariances, priors):
    densities = []
    for i in range(C):
        p = 0
        for j in range(K):
            p += priors[i][j] * MultivariateGaussian(x, means[i][j], covariances[i][j][0])
        densities.append(p)
    Total = 0
    for i in range(C):
        Total += densities[i]
    for i in range(C):
        densities[i] /= Total
    return densities


# In[8]:


def NormalizeDataset(x1, x2):
    x1_mean = sum(x1)/len(x1)
    x2_mean = sum(x2)/len(x2)
    
    x1_var = sum([(xi - x1_mean)**2 for xi in x1])/len(x1)
    x2_var = sum([(xi - x2_mean)**2 for xi in x2])/len(x2)
    
    x1_normal = [(xi - x1_mean)/x1_mean for xi in x1]
    x2_normal = [(xi - x2_mean)/x2_mean for xi in x2]
    return x1_normal, x2_normal


# ## Read and Format Training data,Dev Data

# In[9]:


x1, x2, y = ReadInput('16/train.txt')
y = np.array(y, dtype=int)


# In[10]:


K = 20           # No. of Clusters (for K-Means) or Mixtures (for GMM)
C = len(set(y))  # No. of classes
N = len(y)       # No. of data points

data = []
for i in range(C):
    data.append([])
    
for j in range(C):
    for i in range(len(x1)):
        data[y[i]-1].append([x1[i], x2[i]])
        
entire_data = []
for i in range(len(x1)):
    entire_data.append([x1[i], x2[i]])


# In[11]:


x1_dev, x2_dev, y_dev = ReadInput('16/dev.txt')
y_dev = np.array(y_dev, dtype=int)
N_dev = len(y_dev)
data_dev = []
for i in range(N_dev):
    data_dev.append((x1_dev[i],x2_dev[i]))


# In[12]:


x1_plot_0 = [data[0][i][0] for i in range(len(data[0]))]
x2_plot_0 = [data[0][i][1] for i in range(len(data[0]))]
x1_plot_1 = [data[1][i][0] for i in range(len(data[0]))]
x2_plot_1 = [data[1][i][1] for i in range(len(data[0]))]
plt.scatter(x1_plot_0, x2_plot_0, color='blue')
plt.scatter(x1_plot_1, x2_plot_1, color='orange')
plt.legend(['class 1', 'class 2'])


# # Functions For computing Parameters

# In[13]:


def ComputeParameters(data,K):
    Finalmeans = []
    Finalcovariances = []
    Finalpriors = []
    for j in range(C):
        mean, covariance, prior = InitialEstimate(data[j], K)
        mean, covariance, prior = GMM(data[j], K, prior, mean, covariance)
        Finalpriors.append(prior)
        Finalcovariances.append(covariance)
        Finalmeans.append(mean)
    return Finalmeans,Finalcovariances,Finalpriors


# In[14]:


def ComputeParameters_Diagonal(data,K):
    Finalmeans = []
    Finalcovariances = []
    Finalpriors = []
    for j in range(C):
        mean, covariance, prior = InitialEstimate_Diagonal(data[j], K)
        mean, covariance, prior = GMM_Diagonal(data[j], K, prior, mean, covariance)
        Finalpriors.append(prior)
        Finalcovariances.append(covariance)
        Finalmeans.append(mean)
    return Finalmeans,Finalcovariances,Finalpriors


# ## Plot of clusters and cluster means for K = 3

# In[15]:


means, clusterInfo, freq = KMeans(entire_data, K)
clusterInfo = np.array(clusterInfo, dtype=int)

clusters_x1 = []
clusters_x2 = []
for i in range(K):
    clusters_x1.append([])
    clusters_x2.append([])

for i in range(N):
    clusters_x1[clusterInfo[i]].append(x1[i])
    clusters_x2[clusterInfo[i]].append(x2[i])

for i in range(K):
    plt.scatter(clusters_x1[i], clusters_x2[i])
    plt.scatter(means[i][0], means[i][1])
plt.title('K-means using 20 clusters')


# # Prediction for different k values

# In[16]:


def kVsPrediction(data,data_dev,y_dev,flag):
    kvalues = []
    predictionvalues = []
    for i in range(3,25,4):
        Finalmeans = []
        Finalcovariances = []
        Finalpriors = []
        if(flag==True):  
            Finalmeans,Finalcovariances,Finalpriors = ComputeParameters(data,i)
        else:
            Finalmeans,Finalcovariances,Finalpriors = ComputeParameters_Diagonal(data,i)
        predictionvalues.append(100*classificationPrediction(data_dev,y_dev,C,i,Finalmeans,Finalcovariances,Finalpriors))
    print(predictionvalues)
    return predictionvalues


# k_values = range(3, 25, 4)
# acc_nondiagonal = kVsPrediction(data,data_dev,y_dev,True)
# acc_diagonal = kVsPrediction(data,data_dev,y_dev,False)
# plt.plot(k_values, acc_nondiagonal, marker='o')
# plt.plot(k_values, acc_diagonal, marker='o')
# plt.title('Prediction Accuracy vs K')
# plt.legend(['Non Diagonal Covariances', 'Diagonal Covariances'])
# plt.xlabel('Values of K')
# plt.ylabel('Accuracy')

# In[ ]:





# # Contour plots with different k values

# In[17]:


def creategrid(data):
    if(len(data)==0):
        print('empty data')
    x_list = []
    y_list = []
    for i in range(len(data)):
        x_list.append(data[i][0])
        y_list.append(data[i][1])
    x_list = np.linspace(min(x_list),max(x_list),100)
    y_list = np.linspace(min(y_list),max(y_list),100)
    return x_list,y_list


# In[18]:


tour = 1
def Scatter(x1,x2,data,C,K,Finalmeans,Finalcovariances,Finalpriors):
    fig, ax = plt.subplots(1, 1)
    fig = plt.figure(figsize=(14, 9))

    X_grid1 = np.linspace(min(x1)-tour, max(x1)+tour, 100)
    

    Y_grid1 = np.linspace(min(x2)-tour, max(x2)+tour, 100)
    X, Y = np.meshgrid(X_grid1, Y_grid1)
    
    whichclass = []
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            prob_density = Density([X[i, j], Y[i, j]],C,K,Finalmeans,Finalcovariances,Finalpriors)
            temp = []
            for j in range(C):
                temp.append((prob_density[j],j))
            temp.sort(reverse=True)
            whichclass.append(temp[0][1])
    
    for l in range(2):
        x_list = []
        y_list = []
        count = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if(whichclass[count]==l):
                    x_list.append(X[i,j])
                    y_list.append(Y[i,j])
                count += 1
        if (l==0):
            ax.scatter(x_list,y_list,color = 'yellow')
        elif(l==1):
            ax.scatter(x_list,y_list,color = 'red')
                 
    for i in range(C):
        for j in range(K):
            x_list = np.linspace(Finalmeans[i][j][0]-3,Finalmeans[i][j][0]+3,100)
            y_list = np.linspace(Finalmeans[i][j][1]-3,Finalmeans[i][j][1]+3,100) 
            X_grid , Y_grid = np.meshgrid(x_list, y_list)
            pdf = np.zeros(X_grid.shape)
            for k in range(X_grid.shape[0]):
                for m in range(X_grid.shape[1]):
                    pdf[k, m] = MultivariateGaussian([X_grid[k, m], Y_grid[k, m]], Finalmeans[i][j], Finalcovariances[i][j][0])
            ax.contour(X_grid, Y_grid, pdf,levels = [0.1,0.2,0.3,0.4,0.5],color = 'blue')
    for i in range(C):
        for j in range(len(data[i])):
            if(i==0):
                ax.scatter([data[i][j][0]],[data[i][j][1]],color = 'orange',marker = '.')
            else:
                ax.scatter([data[i][j][0]],[data[i][j][1]],color = 'pink',marker = '.')
    ax.set_title('Contours with k ='+ str(K))
    ax.set_xlabel('x1 dimension')
    ax.set_ylabel('x2 dimension')
    plt.show()


# for k in range(10,26,5):
#     Finalmeans,Finalcovariances,Finalpriors = ComputeParameters(data,k)
#     Scatter(x1,x2,data,C,k,Finalmeans,Finalcovariances,Finalpriors)

# for k in range(10,26,5):
#     Finalmeans,Finalcovariances,Finalpriors = ComputeParameters_Diagonal(data,k)
#     Scatter(x1,x2,data,C,k,Finalmeans,Finalcovariances,Finalpriors)

# # Functions for ROC

# In[19]:


def FindMinMax(S):
    temp = []
    for i in range(len(S)):
        for j in range(len(S[0])):
            temp.append(S[i][j])
    return min(temp),max(temp)


# In[20]:


def ROC(data,c_dev,C,K,Finalmeans,Finalcovariances,Finalpriors):
    S = []
    tests = len(data)
    for i in range(tests):
        prob_density = Density([data[i][0], data[i][1]],C,K,Finalmeans,Finalcovariances,Finalpriors)
        S.append(prob_density)
        #S.append(prob_density[0])
        #S.append(prob_density[1])
    
    Min,Max = FindMinMax(S)
    threshold = np.linspace(Min,Max,100)
    #S.sort()
    #threshold = S

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
        for j in range(tests):
            temp = Density([data[j][0], data[j][1]],C,K,Finalmeans,Finalcovariances,Finalpriors)
            for k in range(C):
                if temp[k] >= i:
                    if c_dev[j] == (k + 1):
                        TP += 1
                    else:
                        FP += 1
                else:
                    if c_dev[j] == (k + 1):
                        FN += 1
                    else:
                        TN += 1
        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))
        FNR.append(FN / (FN + TP))
    return TPR, FPR, FNR


# # Roc curves for different k values

# In[21]:


def RocVsk(data_dev,y_dev,flag):
    fig, ax = plt.subplots(1, 1)
    fig = plt.figure(figsize=(14, 9))
    for k in range(8,26,4):
        Finalmeans = []
        Finalcovariances = []
        Finalpriors = []
        if(flag==True):  
            Finalmeans,Finalcovariances,Finalpriors = ComputeParameters(data,k)
        else:
            Finalmeans,Finalcovariances,Finalpriors = ComputeParameters_Diagonal(data,k)
        TPR, FPR, FNR = ROC(data_dev,y_dev,C,k,Finalmeans,Finalcovariances,Finalpriors)
        ax.plot(FPR, TPR)
    ax.set_title('Roc curves for different values of K')
    ax.set_xlabel('False Positive Rate(FPR)')
    ax.set_ylabel('True Positive Rate(TPR)')
    ax.legend(['k = 8', 'k = 12', 'k = 16', 'k = 20', 'k = 24'])
    #plt.show()


# In[22]:


RocVsk(data_dev,y_dev,True)


# In[23]:


RocVsk(data_dev,y_dev,False)


# # Det for different values of K

# In[24]:


def DetVsk(data_dev,y_dev,flag):
    fig, ax_det = plt.subplots(1, 1)
    fig = plt.figure(figsize=(14, 9))
    for k in range(8,26,4):
        Finalmeans = []
        Finalcovariances = []
        Finalpriors = []
        if(flag==True):  
            Finalmeans,Finalcovariances,Finalpriors = ComputeParameters(data,k)
        else:
            Finalmeans,Finalcovariances,Finalpriors = ComputeParameters_Diagonal(data,k)
        TPR, FPR, FNR = ROC(data_dev,y_dev,C,k,Finalmeans,Finalcovariances,Finalpriors)
        display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)
    ax_det.set_title('Det curves for different values of K')
    ax_det.set_xlabel('False Positive Rate (FPR)')
    ax_det.set_ylabel('False Negative Rate (FNR)')
    plt.show()


# In[25]:


DetVsk(data_dev,y_dev,True)


# DetVsk(data_dev,y_dev,False)

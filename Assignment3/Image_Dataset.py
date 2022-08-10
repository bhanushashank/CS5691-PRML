import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import DetCurveDisplay
import seaborn as sns
from scipy.stats import multivariate_normal 

def MultivariateGaussian(x, mean, cov):
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    constant = 1.0 / ((2 * math.pi) ** (len(x) / 2) * (cov_det ** 0.5)) 
    diff = np.array([x[i] - mean[i] for i in range(len(x))])
    exp = math.exp((-1.0 / 2) * ((diff @ cov_inv) @ diff.T))
    return constant * exp

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
def KMeans(data, K, iteration_limit=20):
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

def classificationPrediction(data,y,C,K,Finalmeans,Finalcovariances,Finalpriors):
    C_Prediction = 0
    N = len(data)
    for i in range(N):
        prob_density = Density(data[i],C,K,Finalmeans,Finalcovariances,Finalpriors)
        temp = []
        for j in range(len(datasets)):
            temp.append((prob_density[j],j))
        temp.sort(reverse=True)
        if(temp[0][1]==(y[i]-1)):
            C_Prediction += 1
    return C_Prediction/N

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

def ComputeParameters(data, K):
    Finalmeans = []
    Finalcovariances = []
    Finalpriors = []
    for j in datasets:
        mean, covariance, prior = InitialEstimate(data[j], K)
        mean, covariance, prior = GMM(data[j], K, prior, mean, covariance)
        Finalpriors.append(prior)
        Finalcovariances.append(covariance)
        Finalmeans.append(mean)
    return Finalmeans,Finalcovariances,Finalpriors

datasets = ['coast', 'forest', 'highway', 'mountain', 'opencountry']   

def ReadInput(subdir):
    data = {scene : [] for scene in datasets}
    for scene in datasets:
        directory = 'Features\\' + scene + '\\' + subdir
        for filename in os.scandir(directory):
            with open(filename.path) as f:
                for line in f:
                    values = list(map(float, line.strip().split(" ")))
                    data[scene].append(values)
    return data

train_data = ReadInput('train')
dev_data = ReadInput('dev')

def Predict(image_data, K, Finalmeans, Finalcovariances, Finalpriors):
    log_likelihood = [0 for i in range(len(datasets))]
    for i in range(36):
        current_log_likelihood = Density(image_data[i], len(datasets), K, Finalmeans, 
                                            Finalcovariances, Finalpriors)
        
        for i in range(len(datasets)):
            log_likelihood[i] += np.log(current_log_likelihood[i])
    log_likelihood = [(log_likelihood[i], datasets[i]) for i in range(len(log_likelihood))]
    unsorted_likelihood = [(log_likelihood[i], datasets[i]) for i in range(len(log_likelihood))]
    log_likelihood.sort()
    return log_likelihood[-1][1], unsorted_likelihood

modified_dev_data = {scene : [] for scene in datasets}

for scene in datasets:
    scene_data = dev_data[scene]
    for i in range(0, len(scene_data), 36):
        current_file = []
        for j in range(36):
            current_file.append(scene_data[i+j])
        modified_dev_data[scene].append(current_file)

accuracy_k = []
log_likelihoods_k = []

for K in range(5, 21, 5):
    correct = 0
    incorrect = 0
    Finalmeans, Finalcovariances, Finalpriors = ComputeParameters(train_data, K)
    #log_likelihoods = []
    for scene in datasets:
        scene_data = modified_dev_data[scene]
        for image_data in scene_data:
            estimated_scene, log_likelihood = Predict(image_data, K, Finalmeans, Finalcovariances, Finalpriors)
            #log_likelihoods.append(log_likelihood)
            if(estimated_scene == scene):
                correct += 1
            else:
                incorrect += 1
    accuracy_k.append(correct/(correct + incorrect))
    #log_likelihoods_k.append(log_likelihoods)
print(accuracy_k)

print(log_likelihoods_k)

def ROC(log_likelihoods, actual_scene):
   
    Thershold = []
    for i in range(len(log_likelihoods)):
        for j in range(len(datasets)):
            Thershold.append(test_results[i][j][0])
    Thershold.sort()

    TPR = []
    FPR = []
    FNR = []
    # TPR = TP/(TP+FN) # True positive rate
    # FPR = FP/(FP+TN) # False positive rate
    for i in Threshold:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(len(log_likelihoods)):
            for k in range(len(datasets)):
                if log_likelihoods[j][k][0] >= i:
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

count = 0
K_range = [5, 10, 15, 20]
TPR_k = []
FPR_k = []
FNR_k = []
for i in range(len(K_range)):
    log_likelihoods = log_likelihoods_k[i]
    TPR, FPR, FNR = ROC(log_likelihoods, actual_class)
    TPR_k.append(TPR)
    FPR_k.append(FPR)
    FNR_k.append(FNR)
    plt.plot(FPR, TPR)
    count += 1
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve for Image Dataset")
plt.legend(["k = 5", "k = 10", "k = 15", "k = 20"])
plt.show()

fig, ax_det = plt.subplots(1, 1)
fig = plt.figure(figsize=(14, 9))
for i in range(len(TPR_k)):
    FPR = FPR_k[i]
    FNR = FNR_k[i]
    display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)
ax_det.set_title('Det curves for Handwriting using DTW')
ax_det.set_xlabel('False Positive Rate (FPR)')
ax_det.set_ylabel('False Negative Rate (FNR)')
plt.show()
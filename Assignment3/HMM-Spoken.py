import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import DetCurveDisplay
import seaborn as sns
import statistics

# File reading

datasets = ['4', '5', '7', '8', 'o']    # datasets assigned to team 16

def ReadInput(subdir):
    #data = {ch : [] for ch in datasets}
    data = [[] for i in range(5)]
    for letter in range(5):
        directory = 'Isolated_Digits/' + datasets[letter] + '/' + subdir
        for filename in os.scandir(directory):
            file_data = []
            if filename.path[-4:] == 'mfcc':
                with open(filename.path) as f:
                    line = f.readline()
                    values = line.strip().split(" ")
                    NC = int(values[0])
                    NF = int(values[1])
                    #file_data.append(NC)
                    #file_data.append(NF)
                    for line in f.readlines():
                        values = list(map(float, line.strip().split(" ")))
                        file_data.append(values)
                data[letter].append(file_data) 
    return data

train_data = ReadInput('train')
dev_data= ReadInput('dev') 

print(train_data[0])
dev_data = []
for i in range(5):
    dev_data.append([])
    temp = len(dev_data_actual[i])
    for j in range(temp):
        dev_data[i].append(dev_data_actual[i][j])

# Helper Functions 

def FindMinMax(S):
    temp = []
    for i in range(len(S)):
        for j in range(len(S[0])):
            temp.append(S[i][j])
    return min(temp),max(temp)

# KMM

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
def KMeans(data, K):
    N = len(data)
    old_clusterInfo = np.zeros(N)    
    new_clusterInfo = np.zeros(N)
    freq = np.zeros(K)
    
    # Pick K random points from the dataset as initial cluster means
    random_indices = random.sample(range(0, N), K)
    means = [np.array(data[idx]) for idx in random_indices]
    
    old_clusterInfo[0] = 1
    
    # Repeat until convergence 
    for ite in range(10):
        old_clusterInfo = new_clusterInfo
        
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

# Roc function

def ROC(likelihood,actual_class,classes):
    Thershold = []
    tests = len(likelihood)
    for i in range(tests):
        for j in range(classes):
            Thershold.append(likelihood[i][j])
    #Min,Max = FindMinMax(Thershold)
    
    Thershold.sort()
    
    TPR = []
    FPR = []
    FNR = []
    # TPR = TP/(TP+FN) # True positive rate
    # FPR = FP/(FP+TN) # False positive rate
    for th in Thershold:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(tests):
            for k in range(classes):
                if likelihood[j][k] >= th:
                    if actual_class[j] == (k + 1):
                        TP += 1
                    else:
                        FP += 1
                else:
                    if actual_class[j] == (k + 1):
                        FN += 1
                    else:
                        TN += 1
        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))
        FNR.append(FN / (FN + TP))
    return TPR, FPR, FNR

# Vector quantization

def vectorquantization(x,mean):
    length = len(mean)
    dis = EuclideanDistance(x,mean[0])
    symbol = 0
    for i in range(1,length):
        temp = EuclideanDistance(x,mean[i])
        if dis > temp:
            dis = temp
            symbol = i
    return symbol

def train_qua(data,classes,mean):
    ans = []
    for i in range(classes):
        ans.append([])
        length = len(data[i])  
        for j in range(length):
            ans[i].append([])
            length1 = len(data[i][j])
            for k in range(length1):
                ans[i][j].append(0)
                
    for i in range(classes):
        length = len(data[i])
        for j in range(length):
            length1 = len(data[i][j])
            for k in range(length1):
                ans[i][j][k] = vectorquantization(data[i][j][k],mean)
    return ans

# Hmm exectution

def hmm_execute(data,classes,nstates,nsymbols,seed,pmin):
    symbol_probs = []
    for k in range(classes):
        symbol_probs.append([])
        for i in range(nstates[k]):
            symbol_probs[k].append([])
        for i in range(nstates[k]):
            symbol_probs[k][i].append([])
            symbol_probs[k][i].append([])
            for j in range(nsymbols):
                symbol_probs[k][i][0].append(0)
                symbol_probs[k][i][1].append(0)
    state_probs = []
    for k in range(classes):
        state_probs.append([])
        for i in range(nstates[k]):
            state_probs[k].append([])
        for i in range(nstates[k]):
            state_probs[k][i].append(0)
            state_probs[k][i].append(0)
    
    for i in range(classes):
        with open("HMM-Code/test.hmm.seq", "w") as f:
            f.write("")
        for file_data in data[i]:
            string = ""
            for symbol in file_data:
                string += str(symbol) + " "
            with open("HMM-Code/test.hmm.seq", "a") as f:
                f.write(string + "\n")    
                    
        os.system(f"cd HMM-Code ; ./train_hmm test.hmm.seq {seed} {nstates[i]} {nsymbols} {pmin}")
        
        with open("HMM-Code/test.hmm.seq.hmm", "r") as f:
            lst = f.readlines()
            count = 1
            for j in range(nstates[i]):
                count += 1
                for k in range(2):
                    #print(lst[count].split("\t"))
                    temp = list(map(float,lst[count].split("\t")[:-1]))
                    count += 1
                    state_probs[i][j][k] = temp[0]
                    for z in range(1,len(temp)):
                        symbol_probs[i][j][k][z-1] = temp[z]
    return state_probs,symbol_probs

def Findlikelihood(x,state_probs,symbol_probs):
    states = len(state_probs)
    symbols = len(symbol_probs[0][0])
    nOb = len(x)
    dp = []
    for i in range(states):
        dp.append([])
        for j in range(nOb+1):
            dp[i].append(0)
    for i in range(states):
        dp[i][0] = 1
    for i in range(1,nOb+1):
        for j in range(states):
            if((j+1) == states):
                dp[j][i] = state_probs[j][0]*symbol_probs[j][0][x[nOb-i]]*dp[j][i-1]
            else:
                dp[j][i] = state_probs[j][0]*symbol_probs[j][0][x[nOb-i]]*dp[j][i-1]
                dp[j][i] += state_probs[j][1]*symbol_probs[j][1][x[nOb-i]]*dp[j+1][i-1]
    return dp[0][symbols]

def covertToArray(data):
    ans = []
    classes = len(data)
    for i in range(classes):
        person = len(data[i])
        for j in range(person):
            length = len(data[i][j])
            for k in range(length):
                ans.append(data[i][j][k])
    return ans

def arrayAndClass(data):
    ans = []
    temp = []
    classes = len(data)
    for i in range(classes):
        person = len(data[i])
        for j in range(person):
            ans.append(data[i][j])
            temp.append(i)
    return ans,temp

# Key execution

def gen(train_data,dev_data,nstates,k):
    temp = covertToArray(train_data)
    means, new_clusterInfo, freq = KMeans(temp, k)
    temp = train_qua(dev_data,5,means)
    data,groundTruth = arrayAndClass(temp)
    temp = train_qua(train_data,5,means)
    state_probs,symbol_probs= hmm_execute(temp,5,nstates,k,543,0.01)
    probs = []
    for i in range(len(data)):
        probs.append([])
        for j in range(5):
            probs[i].append(Findlikelihood(data[i],state_probs[j],symbol_probs[j]))
    return probs,groundTruth

# Confusion matrix

def ConfusionMatrix(data,actual_class,classes):
    N = len(data)
    count = [[0 for i in range(classes)] for j in range(classes)]
    for i in range(N):
        temp = [(data[i][j],j) for j in range(classes)]
        temp.sort()
        count[actual_class[i]][temp[classes-1][1]] += 1
    ax = sns.heatmap(count, annot=True)
    ax.set_xlabel('Actual Class', fontsize=16)
    ax.set_ylabel('Predicted Class', fontsize=16)
    ax.set_title('Confusion Matrix', fontsize=20)
    plt.show()

# Normalization function for handwritting

def Normalization(data):
    classes = len(data)
    for i in range(classes):
        person = len(data[i])
        for j in range(person):
            temp = data[i][j]
            x1 = []
            x2 = []
            for k in range(len(data[i][j])):
                x1.append(data[i][j][k][0])
                x2.append(data[i][j][k][1])
            x1.sort()
            x2.sort()
            diff = [(x1[len(x1)-1]-x1[0]),(x2[len(x2)]-x2[0])]
            mean = [(x1[len(x1)-1]+x1[0])/2,(x2[len(x2)]+x2[0])/2]
            for k in range(len(data[i][j])):
                for m in range(2):
                    data[i][j][k][m] -= mean[m]
                    data[i][j][k][m] /= diff[m]
    return data

# Helper for Roc and Det

def pictureplot(train_data,dev_data,states,k):
    likelihood,groundTruth,state_probs,symbol_probs,pro = gen(train_data,dev_data,states,k)
    ConfusionMatrix(likelihood,groundTruth,5)
    TPR, FPR, FNR = ROC(likelihood,groundTruth,5)
    plt.title('Roc curves at K ='+ str(k))
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.plot(FPR, TPR)
    plt.show()
    fig, ax_det = plt.subplots(1, 1)
    fig = plt.figure(figsize=(14, 9))
    display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)
    ax_det.set_title('Det curves for different values of K')
    ax_det.set_xlabel('False Positive Rate (FPR)')
    ax_det.set_ylabel('False Negative Rate (FNR)')
    plt.show()



def plotRocAndDet():
    states = [[5,5,5,5,5],[5,5,5,5,5],[10,10,10,10,10],[10,10,10,10,10]] 
    k = [50,75,50,75]
    ans = []
    for c in range(len(states)):
        likelihood,groundTruth = gen(train_data,dev_data,states[c],k[c])
        TPR, FPR, FNR = ROC(likelihood,groundTruth,5)
        ans.append([TPR,FPR,FNR])
    
    fig, ax = plt.subplots(1, 1)
    fig = plt.figure(figsize=(14, 9))
    ax.set_title('Roc curves for different values of k and number of states')
    ax.set_xlabel('False Positive Rate(FPR)')
    ax.set_ylabel('True Positive Rate(TPR)')
    for c in range(len(states)):
        ax.plot(ans[c][1],ans[c][0])
    plt.show()
    
    
    fig, ax_det = plt.subplots(1, 1)
    fig = plt.figure(figsize=(14, 9))
    ax_det.set_title('Det curves for different values of k and number of states')
    ax_det.set_xlabel('False Positive Rate (FPR)')
    ax_det.set_ylabel('False Negative Rate (FNR)')
    for c in range(len(states)):
        display = DetCurveDisplay(fpr=ans[c][1], fnr=ans[c][2]).plot(ax=ax_det)
    plt.show()


def listcompare(p1,p2):
    if(len(p1) !=len(p2)):
        return False
    else:
        for i in range(len(p1)):
            if (p1[i]!=p2[i]):
                return False
    return True
    


statesPerClass = [5,5,5,10,10]
k = 75
likelihood,groundTruth = gen(train_data,dev_data,statesPerClass,k)
ConfusionMatrix(likelihood,groundTruth,5)

print(likelihood)

TPR, FPR, FNR = ROC(likelihood,groundTruth,5)

plt.title('Roc curves')
plt.xlabel('False Positive Rate(FPR)')
plt.ylabel('True Positive Rate(TPR)')
plt.plot(FPR, TPR)
plt.show()

fig, ax_det = plt.subplots(1, 1)
fig = plt.figure(figsize=(14, 9))
display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)
ax_det.set_title('Det curves for different values of K')
ax_det.set_xlabel('False Positive Rate (FPR)')
ax_det.set_ylabel('False Negative Rate (FNR)')
plt.show()

statesPerClass = [5,5,5,5,5]
k = 25
pictureplot(train_data,dev_data,statesPerClass,k)

statesPerClass = [5,5,5,5,5]
k = 50
pictureplot(train_data,dev_data,statesPerClass,k)

statesPerClass = [10,10,10,10,10]
k = 25
pictureplot(train_data,dev_data,statesPerClass,k)


statesPerClass = [10,10,10,10,10]
k = 50
pictureplot(train_data,dev_data,statesPerClass,k)

statesPerClass = [5,5,5,10,10]
k = 50
pictureplot(train_data,dev_data,statesPerClass,k)


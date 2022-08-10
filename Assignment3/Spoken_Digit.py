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

datasets = ['4', '5', '7', '8', 'o']    # datasets assigned to team 16
mapping = {'4' : 0 , '5' : 1 , '7' : 2, '8' : 3, 'o' : 4}

def ReadInput(subdir):
    data = {ch : [] for ch in datasets}
    for letter in datasets:
        directory = 'Isolated_Digits\\' + letter + '\\' + subdir
        for filename in os.scandir(directory):
            file_data = []
            if filename.path[-4:] == 'mfcc':
                with open(filename.path) as f:
                    line = f.readline()
                    values = line.strip().split(" ")
                    NC = int(values[0])
                    NF = int(values[1])
                    file_data.append(NC)
                    file_data.append(NF)
                    for line in f.readlines():
                        values = list(map(float, line.strip().split(" ")))
                        file_data.append(values)
                data[letter].append(file_data) 
    return data

train_data = ReadInput('train')
dev_data = ReadInput('dev')

def EuclideanDistance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i])**2
    return math.sqrt(distance)

def DTW(data1, data2):
    
    NC1 = data1[0]
    NF1 = data1[1]
    NC2 = data2[0]
    NF2 = data2[1]
    
    data1 = data1[2:]
    data2 = data2[2:]
    
    n = len(data1)
    m = len(data2)
    dp = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        for j in range(m+1):
            dp[i][j] = 10**15
    dp[0][0] = 0
    w = max(5, abs(n-m))
    for i in range(0, n+1):
        for j in range(max(1, i-w), min(m+1, i+w)):
            dp[i][j] = 0
    
    
    for i in range(1, n+1):
        for j in range(max(1, i-w), min(m+1, i+w)):
            cost = EuclideanDistance(data1[i-1], data2[j-1])
            dp[i][j] = cost + min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1])
    return dp[n][m]

def PickBestFile(data, letter, target, k):
    letter_data = data[letter]
    DTW_values = []
    for i in range(len(letter_data)):
        current_feature_vector = letter_data[i]
        current_DTW = DTW(current_feature_vector, target)
        DTW_values.append(current_DTW)
    DTW_values.sort()
    topK_DTW = DTW_values[:k]
    return topK_DTW

def PickBestLetter(data, target, k):
    DTW_values = []
    average_cost_per_letter = []
    for letter in datasets:
        topK_DTW = PickBestFile(data, letter, target, k)
        current_DTW_avg = sum(topK_DTW)/k
        average_cost_per_letter.append((current_DTW_avg, letter))
    
    average_cost_per_letter_unsorted = [average_cost_per_letter[i] for i in range(len(average_cost_per_letter))]
    average_cost_per_letter.sort()
    optimal_letter = average_cost_per_letter[0][1]
    return optimal_letter, average_cost_per_letter_unsorted

def ConfusionMatrix(data,actual_class,classes):
    N = len(data)
    count = [[0 for i in range(classes)] for j in range(classes)]
    for i in range(N):
        temp = [(data[i][j],j) for j in range(classes)]
        temp.sort()
        count[mapping[actual_class[i]]][temp[classes-1][1]] += 1
    ax = sns.heatmap(count, annot=True)
    ax.set_xlabel('Actual Class', fontsize=16)
    ax.set_ylabel('Predicted Class', fontsize=16)
    ax.set_title('Confusion Matrix', fontsize=20)
    plt.show()

def ROC(test_results, actual_letter):
    Thershold = []
    for i in range(len(test_results)):
        for j in range(len(datasets)):
            Thershold.append(test_results[i][j])
    Thershold.sort()

    TPR = []
    FPR = []
    FNR = []
    # TPR = TP/(TP+FN) # True positive rate
    # FPR = FP/(FP+TN) # False positive rate
    for i in Thershold:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(len(test_results)): 
            for k in range(len(datasets)):
                if test_results[j][k] >= i:
                    if actual_letter[j] == (datasets[k]):
                        TP += 1
                    else:
                        FP += 1
                else:
                    if actual_letter[j] == (datasets[k]):
                        FN += 1
                    else:
                        TN += 1
        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))
        FNR.append(FN / (FN + TP))
    return TPR, FPR, FNR

test_results_k = []
min_max_values = []
actual_class = []
estimated_class_k = []
K_range = [2, 5, 8, 10]

for topk in K_range:    
    success = 0
    total = 0

    test_results = []
    actual_class = []
    estimated_class = []
    for letter in datasets:
        letter_data = dev_data[letter]
        letter_success = 0
        letter_total = 0
        for sample in letter_data:
            estimate, avg_cost_per_letter = PickBestLetter(train_data, sample, topk)
            #print(estimate, letter)
            estimated_class.append(estimate)
            current_results = []
            for j in range(len(datasets)):
                current_results.append(1/avg_cost_per_letter[j][0])
            test_results.append(current_results)
            if estimate == letter:
                success += 1
                letter_success += 1
            total += 1
            letter_total += 1
            actual_class.append(letter)
        print(letter_success, letter_total)
    
    test_results_k.append(test_results)
    estimated_class_k.append(estimated_class)
    print((success/total) * 100)

ConfusionMatrix(test_results,actual_class,5)

TPR_k = []
FPR_k = []
FNR_k = []
count = 0
K_range = [2, 5, 8, 10]
for topk in K_range:
    test_results = test_results_k[count]
    TPR, FPR, FNR = ROC(test_results, actual_class)
    TPR_k.append(TPR)
    FPR_k.append(FPR)
    FNR_k.append(FNR)
    plt.plot(FPR, TPR)
    count += 1
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve for Spoken-Digit Dataset using DTW")
plt.legend(["k = 2", "k = 5", "k = 8", "k = 10"])
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
ax_det.legend(["DTW2", "DTW5", "DTW8", "DTW10"])
plt.show()


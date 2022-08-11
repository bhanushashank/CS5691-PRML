import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import DetCurveDisplay
import seaborn as sns

def UnivariateGaussian(x, mean, var):
    constant = 1.0/(math.sqrt(2*math.pi*var))
    exp = math.exp((-1.0/(2*var)) * ((x - mean)**2))
    return constant * exp


def MultivariateGaussian(x, mean, cov):
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    d = len(x)
    constant = 1.0 / ((2 * math.pi) ** (d / 2) * (cov_det ** 0.5))

    diff = [x[i] - mean[i] for i in range(len(x))]
    diff = np.array(diff)
    exp = math.exp((-1.0 / 2) * ((diff @ cov_inv) @ diff.T))
    return constant * exp

def ReadInput_CSV(file):
    x_input, z_input = [], []
    with open(file) as f:
        for line in f:
            lst = list(map(float, line.split(",")))
            x_input.append([lst[0], lst[1]])
            z_input.append(int(lst[2]))
    x_input = np.array(x_input)
    z_input = np.array(z_input)
    return x_input, z_input

def ClassFrequency(c):
    freq = np.zeros(3)
    for cl in c:
        freq[int(cl)-1] += 1
    freq = freq.astype('int')
    return freq


def MinMax(x):
    X = x[::, 0]
    P = (min(X), max(X))
    Y = x[::, 1]
    Q = (min(Y), max(Y))
    return (P, Q)


def PlotGaussian(x, mean, cov):
    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection='3d')
    my_cmap = plt.get_cmap('cool')

    ans = MinMax(x)
    X_grid = np.linspace(ans[0][0], ans[0][1], 100)
    Y_grid = np.linspace(ans[1][0], ans[1][1], 100)
    X, Y = np.meshgrid(X_grid, Y_grid)
    for k in range(3):
        pdf = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf[i, j] = MultivariateGaussian([X[i, j], Y[i, j]], mean[k], cov[k])
        surf = ax.plot_surface(X, Y, pdf, cmap=my_cmap, edgecolor='none')

    ax.set_title('Probability density curves(PDF)')
    plt.show()


def plotContours(x, mean, cov):
    fig, ax = plt.subplots(1, 1)
    fig = plt.figure(figsize=(14, 9))

    ans = MinMax(x)
    X_grid = np.linspace(ans[0][0], ans[0][1], 100)
    Y_grid = np.linspace(ans[1][0], ans[1][1], 100)
    X, Y = np.meshgrid(X_grid, Y_grid)
    for k in range(3):
        pdf = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf[i, j] = MultivariateGaussian([X[i, j], Y[i, j]], mean[k], cov[k])

        ax.contour(X, Y, pdf)
        e1, e2 = EigenVectors(mean[k], cov[k])
        temp1 = []
        temp2 = []
        for i in range(len(e1)):
            temp1.append(e1[i][0])
            temp2.append(e1[i][1])
        ax.plot(temp1, temp2)
        temp1 = []
        temp2 = []
        for i in range(len(e2)):
            temp1.append(e2[i][0])
            temp2.append(e2[i][1])
        ax.plot(temp1, temp2)

    ax.set_title('Constant density Curves and Eigen Vectors')
    ax.set_xlabel('Feature of x(X1)')
    ax.set_ylabel('Feature of x(X2)')
    plt.show()

def EigenVectors(mean,cov):
    eigenValues,eigenVectors = np.linalg.eig(cov)
    e1 = []
    e2 = []
    helper = np.linspace(-6.0,6.0,10)
    for i in helper:
        temp = [mean[0],mean[1]]
        for j in range(2):
            temp[j] += i*eigenVectors[0][j]
        e1.append(temp)
    for i in helper:
        temp = [mean[0],mean[1]]
        for j in range(2):
            temp[j] += i*eigenVectors[1][j]
        e2.append(temp)
    return e1,e2


def Scatter(x,c):
    for i in range(3):
        X = []
        Y = []
        for j in range(len(x)):
            if(c[j]==i+1):
                X.append(x[j][0])
                Y.append(x[j][1])
        if (i==0):
            plt.plot(X,Y,color = 'green')
        elif(i==1):
            plt.plot(X,Y,color = 'red')
        else:
            plt.plot(X,Y,color = 'blue')

def DecisionBoundary(x, c, mean, cov):
    ans = MinMax(x)
    X_grid = np.linspace(ans[0][0], ans[0][1], 100)
    Y_grid = np.linspace(ans[1][0], ans[1][1], 100)
    X, Y = np.meshgrid(X_grid, Y_grid)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = []
            for k in range(3):
                p.append(MultivariateGaussian([X[i, j], Y[i, j]], mean[k], cov[k]))
            if (p[0] >= p[1] and p[0] >= p[2]):
                plt.plot([X[i, j]], [Y[i, j]], color='green', marker='o')
            elif p[1] >= p[2]:
                plt.plot([X[i, j]], [Y[i, j]], color='red', marker='o')
            else:
                plt.plot([X[i, j]], [Y[i, j]], color='blue', marker='o')
    plt.xlabel('Feature of x(X1)')
    plt.ylabel('Feature of x(X2)')
    plt.title('Decision Boundary and surfaces of three classes')
    for i in range(len(x)):
        if (c[i] == 1):
            plt.plot([x[i][0]], [x[i][1]], color='pink', marker='.')
        elif (c[i] == 2):
            plt.plot([x[i][0]], [x[i][1]], color='yellow', marker='.')
        else:
            plt.plot([x[i][0]], [x[i][1]], color='orange', marker='.')

    plt.show()


def model1(x_train,c_train,mean):
    cov = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(3):
        for j in range(len(x_train)):
            if(c_train[j] == i+1):
                diff = [x_train[j][0] - mean[i][0], x_train[j][1] - mean[i][1]]
                diff = np.array(diff)
                temp = [[0,0],[0,0]]
                temp[0][0] = diff[0]*diff[0]
                temp[0][1] = diff[0]*diff[1]
                temp[1][0] = diff[0]*diff[1]
                temp[1][1] = diff[1]*diff[1]
                for k in range(2):
                    for l in range(2):
                        cov[k][l] += temp[k][l]


    for i in range(2):
        for j in range(2):
            cov[i][j] /= (len(x_train)-3)
    return [cov,cov,cov]

def model2(x_train,c_train,mean,class_frequencies):
    cov = []
    for i in range(3):
        temp1 = [[0,0],[0,0]]
        for j in range(len(x_train)):
            if(c_train[j] == i+1):
                diff = [x_train[j][0] - mean[i][0], x_train[j][1] - mean[i][1]]
                diff = np.array(diff)
                temp = [[0,0],[0,0]]
                temp[0][0] = diff[0]*diff[0]
                temp[0][1] = diff[0]*diff[1]
                temp[1][0] = diff[0]*diff[1]
                temp[1][1] = diff[1]*diff[1]
                for k in range(2):
                    for l in range(2):
                        temp1[k][l] += temp[k][l]
        for j in range(2):
            for k in range(2):
                temp1[j][k] /= (class_frequencies[i]-1)
        cov.append(temp1)
    return cov

def model3(x_train,c_train,mean):
    sum = 0
    for i in range(len(x_train)):
        sum += (x_train[i][0]-mean[c_train[i]-1][0])**2
        sum += (x_train[i][1]-mean[c_train[i]-1][1])**2

    sum /= len(x_train)

    cov = [[0.0,0.0],[0.0,0.0]]
    for i in range(2):
        cov[i][i] = sum
    return [cov,cov,cov]

def model4(x_train,c_train,mean):
    diagonal_elements = [0.0,0.0]
    for i in range(2):
        sum = 0
        for j in range(len(x_train)):
            sum += (x_train[j][i]-mean[c_train[j]-1][i])**2
        diagonal_elements[i] = sum/len(x_train)

    cov = [[0.0,0.0],[0.0,0.0]]
    for i in range(2):
        cov[i][i] = diagonal_elements[i]
    return [cov,cov,cov]

def model5(x_train,c_train,mean,class_frequencies):
    cov = []
    for i in range(3):
        diagonal_elements = [0.0,0.0]
        for j in range(len(x_train)):
                if(c_train[j] == i+1):
                    diagonal_elements[0] += (x_train[j][0]-mean[i][0])**2
                    diagonal_elements[1] += (x_train[j][1]-mean[i][1])**2
        for j in range(2):
            diagonal_elements[j] /= class_frequencies[i]
        temp = [[0.0,0.0],[0.0,0.0]]
        for j in range(2):
            temp[j][j] = diagonal_elements[j]
        cov.append(temp)
    return cov


def ROC(x_dev, c_dev, mean, cov):
    S = []
    for i in range(len(x_dev)):
        for j in range(3):
            temp = MultivariateGaussian(x_dev[i], mean[j], cov[j])
            S.append(temp)
    S.sort()

    TPR = []
    FPR = []
    FNR = []
    # TPR = TP/(TP+FN) # True positive rate
    # FPR = FP/(FP+TN) # False positive rate
    for i in S:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(len(x_dev)):
            for k in range(3):
                temp = MultivariateGaussian(x_dev[j], mean[k], cov[k])
                if temp >= i:
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

def ConfusionMatrix(x_dev,c_dev,mean,cov):
    count = np.zeros((3,3))
    for i in range(len(x_dev)):
        p = []
        for j in range(3):
            p.append(MultivariateGaussian(x_dev[i],mean[j],cov[j]))
        if p[0]>=p[1] and p[0] >= p[2]:
            count[c_dev[i]-1][0] += 1
        elif p[1]>=p[2]:
            count[c_dev[i]-1][1] += 1
        else:
            count[c_dev[i]-1][2] += 1
    return count


def BayesClassifier(x_train, c_train, x_dev, c_dev):
    class_frequencies = ClassFrequency(c_train)
    prior_probability = []
    mean = []
    covariance = []

    n = len(c_train)
    for i in range(3):
        prior_probability.append(class_frequencies[i] / n)
        temp = [0, 0]
        count = 0
        for j in range(n):
            if (c_train[j] == i + 1):
                temp[0] += x_train[j][0]
                temp[1] += x_train[j][1]
                count += 1
        if (count > 0):
            mean.append([temp[0] / count, temp[1] / count])

    covariance.append(model1(x_train, c_train, mean))
    covariance.append(model2(x_train, c_train, mean, class_frequencies))
    covariance.append(model3(x_train, c_train, mean))
    covariance.append(model4(x_train, c_train, mean))
    covariance.append(model5(x_train, c_train, mean, class_frequencies))

    for i in range(5):
        PlotGaussian(x_train, mean, covariance[i])
        plotContours(x_train, mean, covariance[i])
        DecisionBoundary(x_train, c_train, mean, covariance[i])
        count = ConfusionMatrix(x_dev, c_dev, mean, covariance[4])
        ax = sns.heatmap(count, annot=True)
        ax.set_xlabel('Actual Class', fontsize=16)
        ax.set_ylabel('Predicted Class', fontsize=16)
        ax.set_title('Confusion Matrix', fontsize=20)
        plt.show()

    TPR, FPR, FNR = ROC(x_dev, c_dev, mean, covariance[0])
    plt.plot(FPR, TPR)
    TPR, FPR, FNR = ROC(x_dev, c_dev, mean, covariance[1])
    plt.plot(FPR, TPR)
    TPR, FPR, FNR = ROC(x_dev, c_dev, mean, covariance[2])
    plt.plot(FPR, TPR)
    TPR, FPR, FNR = ROC(x_dev, c_dev, mean, covariance[3])
    plt.plot(FPR, TPR)
    TPR, FPR, FNR = ROC(x_dev, c_dev, mean, covariance[4])
    plt.plot(FPR, TPR)
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('ROC Curves')
    plt.legend(["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"])
    plt.show()

    fig, ax_det = plt.subplots(1, 1, figsize=(5, 5))
    TPR, FPR, FNR = ROC(x_dev, c_dev, mean, covariance[0])
    display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)
    TPR, FPR, FNR = ROC(x_dev, c_dev, mean, covariance[1])
    display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)
    TPR, FPR, FNR = ROC(x_dev, c_dev, mean, covariance[2])
    display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)
    TPR, FPR, FNR = ROC(x_dev, c_dev, mean, covariance[3])
    display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)
    TPR, FPR, FNR = ROC(x_dev, c_dev, mean, covariance[4])
    display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('False Negetive Rate(FNR)')
    plt.title('DET Curves')
    plt.legend(["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"])
    plt.show()

x_train_input, c_train_input = ReadInput_CSV('linear_train.txt')
x_dev_input, c_dev_input = ReadInput_CSV('linear_dev.txt')

x_train, c_train = np.array(x_train_input), np.array(c_train_input)
x_dev  , c_dev = np.array(x_dev_input), np.array(c_dev_input)

BayesClassifier(x_train,c_train, x_dev, c_dev)


x_train_input, c_train_input = ReadInput_CSV('nonlinear_train.txt')
x_dev_input, c_dev_input = ReadInput_CSV('nonlinear_dev.txt')

x_train, c_train = np.array(x_train_input), np.array(c_train_input)
x_dev  , c_dev = np.array(x_dev_input), np.array(c_dev_input)

BayesClassifier(x_train,c_train, x_dev, c_dev)

x_train_input, c_train_input = ReadInput_CSV('trian.txt')
x_dev_input, c_dev_input = ReadInput_CSV('dev.txt')

x_train, c_train = np.array(x_train_input), np.array(c_train_input)
x_dev  , c_dev = np.array(x_dev_input), np.array(c_dev_input)

BayesClassifier(x_train,c_train, x_dev, c_dev)
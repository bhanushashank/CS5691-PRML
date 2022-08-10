import math
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.ticker as ticker

def ReadInput(file, flag=False):
    x_input, y_input, z_input = [], [], []
    with open(file) as f:
        for line in f:
            lst = list(map(float, line.split(" ")))
            x_input.append(lst[0])
            y_input.append(lst[1])
            if(flag):
                z_input.append(lst[2])
    x_input = np.array(x_input)
    y_input = np.array(y_input)
    z_input = np.array(z_input)
    if flag:
        return x_input, y_input, z_input
    return x_input, y_input

# Generates a random sample of the given size
def GenerateSample(x_input, y_input, sampleSize):
    sample_indices = np.random.choice(range(len(x_input)), size=sampleSize)
    x_input_sample, y_input_sample = x_input[sample_indices], y_input[sample_indices]
    x_input_sample, y_input_sample = np.array(x_input_sample), np.array(y_input_sample)
    sorted_indices = x_input_sample.argsort()[::]
    x_input_sample, y_input_sample = x_input_sample[sorted_indices], y_input_sample[sorted_indices]
    return x_input_sample, y_input_sample

# [1, x, x^2, .., x^degree]
def CreateBasis(x,degree):
    current = 1
    output = []
    for i in range(degree+1):
        output.append(current)
        current *= x
    return output

# Returns output vector given basis and weight vectors
def Multiply(weights, basis):
    n = len(weights)
    assert(n == len(basis))
    output = 0
    for i in range(n):
        output += weights[i] * basis[i]
    return output


# Generates phi matrix
def CreatePhi(x_train, degree):
    phi = np.zeros((len(x_train), degree+1))
    for i in range(len(x_train)):
        phi[i] = CreateBasis(x_train[i], degree)
    return phi

# Returns Pseudo Inverse of matrix A
def PsuedoInverse(A, lam=0):
    AT = A.T
    ATA = AT @ A 
    lam_I = np.zeros(ATA.shape)
    for i in range(ATA.shape[0]):
        lam_I[i][i] = lam
    ATA = ATA + lam_I
    ATA_Inv = np.linalg.inv(ATA)
    return ATA_Inv @ AT

def ComputeWeights(x_train, y_train, degree, lam=0):
    phi = CreatePhi(x_train, degree)
    phiPlus = PsuedoInverse(phi, lam)
    weights = phiPlus @ y_train
    return weights
    
# Gives predicted y values 
def Predict(x_train, weights, degree):
    output = []
    for i in range(len(x_train)):
        basis = CreateBasis(x_train[i], degree)
        pred  = Multiply(weights, basis)
        output.append(pred)
    return output

# Performs Linear (Least Square) Regression 
def LinearRegression(x_train, y_train, degree, lam=0):
    #print(lam)
    weights = ComputeWeights(x_train, y_train, degree, lam)    
    output = Predict(x_train, weights, degree)
    return output, weights

# Computes the sum-of-squares error
def SquaredError(estimated, actual, weights, lam=0):
    assert(len(estimated) == len(actual))
    error = 0
    norm = 0
    for i in range(len(estimated)):
        error += (estimated[i] - actual[i]) ** 2
    for x in weights:
        norm += x*x
    return (error + lam*norm)/2

# Plotting function
def Plot2D(x_values, y_values, x_label="", y_label="", marker=[""], title="", Legend=[]):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    for i in range(len(x_values)):
        plt.plot(x_values[i], y_values[i], marker=marker[max(i, 0)])
        
    if len(Legend) > 0:
        plt.legend(Legend)
    plt.show()
    
# Plotting function for Scatter Plots
def Plot2D_Scatter(x_values, y_values, x_label="", y_label="", marker=[""], title="", Legend=[], hollow=False):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    for i in range(len(x_values)):
        if hollow:
            plt.scatter(x_values[i], y_values[i], marker=marker[max(i, 0)], facecolors='none', edgecolors='b')
        else:
            plt.scatter(x_values[i], y_values[i], marker=marker[max(i, 0)])
        
    if len(Legend) > 0:
        plt.legend(Legend)
    plt.show()

def TrainSample(sampleSize, degree):
    x_train, y_train = GenerateSample(x_train_input, y_train_input, sampleSize)
    regression_estimate, weights = LinearRegression(x_train, y_train, degree)
    return x_train, y_train, regression_estimate

x_train_input, y_train_input = ReadInput('1d_team_16_train.txt')
x_dev_input, y_dev_input = ReadInput('1d_team_16_dev.txt')

plt.scatter(x_train_input, y_train_input, facecolors='none', edgecolors='b')
plt.title("Training Data")
plt.show()

plt.scatter(x_dev_input, y_dev_input, facecolors='none', edgecolors='b')
plt.title("Development Data")
plt.show()

model_complexity = [x for x in range(1, 8)]
model_error_train = []
model_error_dev = []
for i in range(1, 8):
    x_train,y_train= GenerateSample(x_train_input,y_train_input,200)
    regression_estimate, weights = LinearRegression(x_train, y_train, i)
    error_train = SquaredError(regression_estimate, y_train,weights)
    model_error_train.append(error_train)
    
    x_dev,y_dev = GenerateSample(x_dev_input,y_dev_input,200)
    
    estimate_dev = Predict(x_dev, weights, i)
    error_dev = SquaredError(estimate_dev, y_dev, weights)
    model_error_dev.append(error_dev)

degree = 7
sampleSize = 200
x_train_sample, y_train_sample, regression_estimate_sample= TrainSample(sampleSize, degree)

Plot2D([x_train_sample, x_train_sample], [regression_estimate_sample, y_train_sample], 
       x_label="Values of x", y_label ="Values of y", marker=["*", "o"], 
       title="Estimated vs Actual (Training Data, degree=" + str(degree) + ", sample size=" + str(sampleSize) + ")", 
       Legend=["Estimate", "Original"])

Plot2D_Scatter([regression_estimate_sample], [y_train_sample], x_label="Estimated values from the model", y_label ="Actual data values", 
       marker=["o"], title="Scatter Plot of estimated vs actual values on training data, sample size = " + str(sampleSize), Legend=[], hollow=True)

# error plot on development data for various model complexities
Plot2D([model_complexity], [model_error_train], x_label="Degree of the polynomial", y_label ="Error of the model", 
       marker=["*", "o"], title="Error vs complexity plot for Training data", Legend=[])


choices = range(1, 8, 2)

idx = 0
rows = 2
cols = 2
fig, axs = plt.subplots(rows, cols, figsize=(12, 12))

for i in range(rows):
    for j in range(cols):
        regression_estimate, weights = LinearRegression(x_train, y_train, choices[idx])
        axs[i, j].plot(x_train, regression_estimate,color = 'r')
        axs[i, j].scatter(x_train, y_train, facecolors='none', edgecolors='b')
        axs[i, j].legend(["Estimate", "Actual Data"])
        axs[i, j].set_title("Degree of polynomial = " + str(choices[idx]))
        idx += 1
        
fig.suptitle("Comparison of solution curves for different model complexities",fontsize=15)
fig.tight_layout()
plt.show()


plt.scatter(x_train_sample, y_train_sample, facecolors='none', edgecolors='b')
plt.plot(x_train_sample, regression_estimate_sample, color="red")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Solution plot (red curve) for training data (blue points). Sample size = " + str(sampleSize) + ", Degree = " + str(degree))
plt.show()


degree = 7
sampleSize = 200
regression_estimate, weights = LinearRegression(x_train_sample, y_train_sample, degree)
x_dev, y_dev = GenerateSample(x_dev_input, y_dev_input, sampleSize)
estimate_dev = Predict(x_train_sample, weights, degree)
Plot2D([x_dev, x_dev], [estimate_dev, y_dev], x_label="Values of x", y_label ="Values of y", 
       marker=["*", "o"], title="Estimated vs Actual (Development Data, degree=" + str(degree) + 
       ", sample size=" + str(sampleSize) + ")", Legend=["Estimate", "Original"])

Plot2D_Scatter([estimate_dev], [y_dev], x_label="Estimated values from the model", y_label ="Actual data values", 
       marker=["o"], title="Scatter Plot of estimated vs actual values on development data", Legend=[])

# error plot on development data for various model complexities
Plot2D([model_complexity], [model_error_dev], x_label="Degree of the polynomial", y_label ="Error of the model", 
       marker=["*", "o"], title="Error vs complexity plot for development data", Legend=[])


# error plot on development data for various model complexities
Plot2D([model_complexity, model_complexity], [model_error_dev, model_error_train], x_label="Degree of the polynomial", y_label ="Error of the model", 
       marker=["*", "o"], title="Error vs Model Complexity", Legend=["Development", "Training"])

sampleSize = 20
degree = 7
x_train_sample, y_train_sample, regression_estimate_sample = TrainSample(sampleSize, degree)
plt.scatter(x_train_sample, y_train_sample, facecolors='none', edgecolors='b')
plt.plot(x_train_sample, regression_estimate_sample, color="red")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Sample size = " + str(sampleSize) + ", Degree = " + str(degree))
plt.show()

sampleSize = 50
degree = 7
x_train_sample, y_train_sample, regression_estimate_sample = TrainSample(sampleSize, degree)
plt.scatter(x_train_sample, y_train_sample, facecolors='none', edgecolors='b')
plt.plot(x_train_sample, regression_estimate_sample, color="red")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Sample size = " + str(sampleSize) + ", Degree = " + str(degree))
plt.show()

sampleSize = 100
degree = 7
x_train_sample, y_train_sample, regression_estimate_sample = TrainSample(sampleSize, degree)
plt.scatter(x_train_sample, y_train_sample, facecolors='none', edgecolors='b')
plt.plot(x_train_sample, regression_estimate_sample, color="red")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Sample size = " + str(sampleSize) + ", Degree = " + str(degree))
plt.show()

sampleSize = 150 
degree = 7
x_train_sample, y_train_sample, regression_estimate_sample = TrainSample(sampleSize, degree)
plt.scatter(x_train_sample, y_train_sample, facecolors='none', edgecolors='b')
plt.plot(x_train_sample, regression_estimate_sample, color="red")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Sample size = " + str(sampleSize) + ", Degree = " + str(degree))
plt.show()

sampleSize = 200 
degree = 7
x_train_sample, y_train_sample, regression_estimate_sample = TrainSample(sampleSize, degree)
log_lam_values = range(-20, 0, 2)
 
EM = []
EM_dev = []
for i in range(len(log_lam_values)):
    regression_estimate, weights = LinearRegression(x_train, y_train, 7, math.exp(log_lam_values[i]))
    error = SquaredError(regression_estimate, y_train, weights, math.exp(log_lam_values[i]))
    EM.append(error)
    
    estimate_dev = Predict(x_dev, weights, 7)
    error_dev = SquaredError(estimate_dev, y_dev, weights)
    EM_dev.append(error_dev)
    
plt.xlabel("ln(lambda)") 
plt.ylabel("Squared Error")
plt.plot(log_lam_values, EM, marker='o', color = 'red')
plt.plot(log_lam_values, EM_dev, marker='o', color = 'blue')
plt.legend(["Training", "Development"])
plt.title("Error vs Regularization Parameter for 1D data")
plt.show()



########### 2D Data ###########
# Reads and returns two dimensional data from a file
def ReadInput2D(file):
    x1_input, x2_input, z_input = [], [], []
    with open(file) as f:
        for line in f:
            lst = list(map(float, line.split(" ")))
            x1_input.append(lst[0])
            x2_input.append(lst[1])
            z_input.append(lst[2])
    x1_input = np.array(x1_input)
    x2_input = np.array(x2_input)
    z_input = np.array(z_input)
    return x1_input, x2_input, z_input

# Generates a random sample of the given size
def GenerateSample2D(x_input, y_input, z_input, sampleSize):
    sample_indices = np.random.choice(range(len(x_input)), size=sampleSize)
    x_input_sample, y_input_sample, z_input_sample = x_input[sample_indices], y_input[sample_indices], z_input[sample_indices]
    x_input_sample, y_input_sample, z_input_sample = np.array(x_input_sample), np.array(y_input_sample), np.array(z_input_sample)
    sorted_indices = x_input_sample.argsort()[::]
    x_input_sample, y_input_sample, z_input_sample = x_input_sample[sorted_indices], y_input_sample[sorted_indices], z_input_sample[sorted_indices]
    return x_input_sample, y_input_sample, z_input_sample

# Create Basis in two dimensions, consisting of terms of all possible degrees 1 and x^degree * y^degree
def CreateBasis2D(x, y, degree):
    term = 1
    basis = []

    for i in range(degree+1):
        term_y = term
        for j in range(degree+1):
            basis.append(term_y)
            term_y *= y
        term *= x 
    return basis

# Generates phi matrix
def CreatePhi2D(x_train, y_train, degree):
    phi = np.zeros((len(x_train), (degree+1)**2))
    for i in range(len(x_train)):
        phi[i] = CreateBasis2D(x_train[i], y_train[i], degree)
    return phi

# Determines the weight vector using MLE
def ComputeWeights2D(x_train, y_train, z_train, degree, lam=0):
    phi = CreatePhi2D(x_train, y_train, degree)
    phiPlus = PsuedoInverse(phi, lam)
    weights = phiPlus @ z_train
    return weights
    
# Gives output (y) values predicted by the model
def Predict2D(x_train, y_train, weights, degree):
    output = []
    for i in range(len(x_train)):
        basis = CreateBasis2D(x_train[i],y_train[i], degree)
        pred  = Multiply(weights, basis)
        output.append(pred)
    return output

# Performs Linear Regression for 2D data
def LinearRegression2D(x_train, y_train, z_train, degree, lam=0):
    weights = ComputeWeights2D(x_train, y_train,z_train, degree, lam)    
    output  = Predict2D(x_train, y_train, weights, degree)
    return output, weights

# Creates a model of the given degree using training sample of the specified size
def TrainSample2D(sampleSize, degree):
    x_train, y_train, z_train    = GenerateSample2D(x_train_input, y_train_input,z_train_input, sampleSize)
    regression_estimate, weights = LinearRegression2D(x_train, y_train,z_train, degree)
    return x_train, y_train, z_train, regression_estimate

# Plotting function
def Plot2D(x_values, y_values, x_label="", y_label="", marker=[""], title="", Legend=[]):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    for i in range(len(x_values)):
        plt.plot(x_values[i], y_values[i], marker=marker[max(i, 0)])
    if len(Legend) > 0 and Legend[0] != "":
        plt.legend(Legend)
    plt.show()
    
# Plotting function for Scatter Plots
def Plot2D_Scatter(x_values, y_values, x_label="", y_label="", marker=[""], title="", Legend=[], hollow=False):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    for i in range(len(x_values)):
        if hollow:
            plt.scatter(x_values[i], y_values[i], marker=marker[max(i, 0)], facecolors='none', edgecolors='b')
        else:
            plt.scatter(x_values[i], y_values[i], marker=marker[max(i, 0)])
        
    if len(Legend) > 0:
        plt.legend(Legend)
    plt.show()

# Read training data and development data from the given datasets
x_train_input, y_train_input, z_train_input = ReadInput2D('2d_team_16_train.txt')
x_dev_input, y_dev_input, z_dev_input       = ReadInput2D('2d_team_16_dev.txt')

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
ax.scatter3D(x_train_input, y_train_input, z_train_input, color = "blue")
plt.title("Training Data")

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection = "3d")
ax.scatter3D(x_dev_input, y_dev_input, z_dev_input, color = "blue")
plt.title("Development Data")


x_train_sample, y_train_sample, z_train_sample = GenerateSample2D(x_train_input, y_train_input, z_train_input, 1000)
x_dev_sample, y_dev_sample, z_dev_sample = GenerateSample2D(x_dev_input, y_dev_input, z_dev_input, 1000)

choices = range(1, 8, 2)

idx = 0
rows = 2
cols = 2
fig, axs = plt.subplots(rows, cols, figsize=(12, 12))

idx=0
for i in range(rows):
    for j in range(cols):
        regression_estimate, weights = LinearRegression2D(x_train_sample, y_train_sample, z_train_sample, choices[idx])
        ax = fig.add_subplot(221+idx, projection='3d')
        ax.scatter(x_train_sample, y_train_sample, regression_estimate,  marker="o", c="blue")
        ax.scatter(x_train_sample, y_train_sample, z_train_sample,  marker="o", c="red")
        idx += 1
        
fig.suptitle("Comparison of solution curves for different model complexities",fontsize=15)
fig.tight_layout()
plt.show()

model_complexity = [x for x in range(1, 8)]
model_error_train = []
model_error_dev = []
for i in range(1, 8):
    x_train, y_train, z_train = GenerateSample2D(x_train_input,y_train_input,z_train_input,1000)
    regression_estimate, weights = LinearRegression2D(x_train, y_train, z_train, i)
    error_train = SquaredError(regression_estimate, z_train,weights)
    model_error_train.append(error_train)
     
    x_dev,y_dev,z_dev = GenerateSample2D(x_dev_input,y_dev_input,z_dev_input,1000)
    
    estimate_dev = Predict2D(x_dev,y_dev, weights, i)
    error_dev = SquaredError(estimate_dev,z_dev, weights)
    model_error_dev.append(error_dev)



degree = 7
sampleSize = 1000
x_train_sample, y_train_sample, z_train_sample, regression_estimate_sample = TrainSample2D(sampleSize, degree)

Plot2D_Scatter([regression_estimate_sample], [z_train_sample], x_label="Estimated values from the model", y_label ="Actual data values", 
       marker=["o"], title="Scatter Plot of estimated vs actual values on training data, sample size = " + str(sampleSize), Legend=[])

# error plot on development data for various model complexities
Plot2D([model_complexity], [model_error_train], x_label="Degree of the polynomial", y_label ="Error of the model", 
       marker=["*", "o"], title="Error vs complexity plot for Training data", Legend=[])

regression_estimate, weights = LinearRegression2D(x_train_sample, y_train_sample,z_train_sample, degree)
x_dev, y_dev,z_dev = GenerateSample2D(x_dev_input, y_dev_input,z_dev_input, sampleSize)
estimate_dev = Predict2D(x_train_sample,y_train_sample, weights, degree)

Plot2D_Scatter([estimate_dev], [z_dev], x_label="Estimated values from the model", y_label ="Actual data values", 
       marker=["o"], title="Scatter Plot of estimated vs actual values on development data", Legend=[])

# error plot on development data for various model complexities
Plot2D([model_complexity], [model_error_dev], x_label="Degree of the polynomial", y_label ="Error of the model", 
       marker=["*", "o"], title="Error vs complexity plot for development data", Legend=[])


plt.plot(model_complexity, model_error_train, color="blue", marker="o")
plt.plot(model_complexity, model_error_dev, color="red", marker="o")
plt.xlabel("Degree of Polynomial (Model Complexity)")
plt.ylabel("Squared Error")
plt.title("Squared Error vs Model Complexity for 2D data")
plt.legend(["Training Data", "Development Data"])
plt.show()

log_lam_values = range(-40, -2, 2)
 
#x_train, y_train = GenerateSample(x_train_input, y_train_input, 200)
EM = []
EM_dev = []
for i in range(len(log_lam_values)):
    regression_estimate, weights = LinearRegression2D(x_train, y_train, z_train, 7, math.exp(log_lam_values[i]))
    
    error = SquaredError(regression_estimate, z_train, weights,math.exp(log_lam_values[i]))
    EM.append(error)
    x_dev,y_dev,z_dev = GenerateSample2D(x_dev_input,y_dev_input,z_dev_input, 1000)
    estimate_dev = Predict2D(x_dev,y_dev, weights, 7)
    error_dev = SquaredError(estimate_dev,z_dev, weights)
    EM_dev.append(error_dev)
    
plt.xlabel("ln(Lambda)") 
plt.ylabel("Regression Error")
plt.plot(log_lam_values, EM, marker='o', color = 'red')
plt.legend(["Training"])
plt.show()

plt.xlabel("ln(Lambda)") 
plt.ylabel("Regression Error")
plt.plot(log_lam_values, EM_dev, marker='o', color = 'red')
plt.legend(["Development Data"])
plt.show()
# a logistic regression classifier to recognize cats

## 1. Packages

```
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
```

## 2. pre-processing new dataset

Common steps for pre-processing a new dataset are:

- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
- Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
- "Standardize" the data

## 3. General Architecture of the learning algorithm 

[image](http://om0ttwn6c.bkt.clouddn.com/LogReg_kiank.png)

- Initialize the parameters of the model
- Learn the parameters for the model by minimizing the cost  
- Use the learned parameters to make predictions (on the test set)
- Analyse the results and conclude

## 4. Building the parts of our algorithm

### 4.1 Helper functions

```
#sigmoid函数
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s
```

### 4.2 Initializing parameters

```
#初始化参数w,b
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros((dim,1)) #w为一个dim*1矩阵
    b = 0    
    return w, b
```

### 4.3 Computes the cost function and its gradient

```
#计算Y_hat,成本函数J以及dw，db
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """
    m = X.shape[1] #样本个数
    Y_hat = sigmoid(np.dot(w.T,X)+b)                                     
    cost = -(np.sum(np.dot(Y,np.log(Y_hat).T)+np.dot((1-Y),np.log(1-Y_hat).T)))/m #成本函数
    dw = (np.dot(X,(Y_hat-Y).T))/m
    db = (np.sum(Y_hat-Y))/m
    cost = np.squeeze(cost) #压缩维度    
    grads = {"dw": dw,
             "db": db} #梯度
    return grads, cost 
```

### 4.4 Optimization

```
#梯度下降找出最优解
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):#num_iterations-梯度下降次数 learning_rate-学习率，即参数ɑ
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    
    costs = [] #记录成本值
    for i in range(num_iterations): #循环进行梯度下降
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate*dw
        b = b - learning_rate*db
        if i % 100 == 0: #每100次记录一次成本值
            costs.append(cost)
        if print_cost and i % 100 == 0: #打印成本值
            print ("循环%i次后的成本值: %f" %(i, cost))
    params = {"w": w,
              "b": b} #最终参数值
    grads = {"dw": dw,
             "db": db}#最终梯度值
    return params, grads, costs
    
```

### 4.5 Predict

```
#预测出结果
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1] #样本个数
    Y_prediction = np.zeros((1,m)) #初始化预测输出
    w = w.reshape(X.shape[0], 1) #转置参数向量w
    Y_hat = sigmoid(np.dot(w.T,X)+b) #最终得到的参数代入方程
    for i in range(Y_hat.shape[1]):
        if Y_hat[:,i]>0.5:
            Y_prediction[:,i] = 1
        else:
            Y_prediction[:,i] = 0
    return Y_prediction
    
```

## 5. Merge all functions into a model

### 5.1 model 

- Preprocessing the dataset is important.
- You implemented each function separately: initialize(), propagate(), optimize(). Then you built a model().
- Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference to the algorithm.

```
#建立整个预测模型
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False): #num_iterations-梯度下降次数 learning_rate-学习率，即参数ɑ
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    w, b = initialize_with_zeros(X_train.shape[0]) #初始化参数w，b
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost) #梯度下降找到最优参数
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_train = predict(w, b, X_train) #训练集的预测结果
    Y_prediction_test = predict(w, b, X_test) #测试集的预测结果
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100 #训练集识别准确度
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100 #测试集识别准确度
    print("训练集识别准确度: {} %".format(train_accuracy))
    print("测试集识别准确度: {} %".format(test_accuracy))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d
    
```

### 5.2 Run Model

```
#初始化数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0] #训练集中样本个数
m_test = test_set_x_orig.shape[0] #测试集总样本个数
num_px = test_set_x_orig.shape[1] #图片的像素大小
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T #原始训练集的设为（12288*209）
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T #原始测试集设为（12288*50）
train_set_x = train_set_x_flatten/255. #将训练集矩阵标准化
test_set_x = test_set_x_flatten/255. #将测试集矩阵标准化
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
```

## 6. mathplotlib

```
# 画出学习曲线
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
```

```
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("学习率: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')
for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
plt.ylabel('cost')
plt.xlabel('iterations')
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
```


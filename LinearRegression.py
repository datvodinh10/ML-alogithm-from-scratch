from sklearn import datasets
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
X,y = datasets.load_boston(return_X_y=True)

X_train,X_test = X[:500],X[500:]
y_train,y_test = y[:500],y[500:]
X_test_new = (X_test-X_train.min())/(X_train.max()-X_train.min())
X_train_new = (X_train-X_train.min())/(X_train.max()-X_train.min())
y_test_new =  (y_test - y_train.min()) / (y_train.max()-y_train.min())
y_train_new  = (y_train - y_train.min()) / (y_train.max()-y_train.min())


def reversed(y_pred):
    return y_pred * (y_train.max()-y_train.min()) + y_train.min()
y_test,y_test_new

class MyLinearRegression() :
    def __init__(self, X, y, alpha = 0.1,gamma = 0.9) :
        self.X = X
        self.y = np.array([y]).T
        self.N = self.X.shape[0]
        ones = np.ones((self.N, 1))
        self.alpha = alpha
        self.gamma = gamma
        self.Xbar = np.concatenate((self.X, ones), axis= 1)
        self.theta = np.random.randn((self.X.shape[1]+1), 1)
        self.v0 = 0
    def y_hat(self) :
        return self.Xbar @ self.theta
    
    # def standardization(self,X,y):
    #     self.X = (self.X-self.
    def cost_function(self):      
        return np.mean((self.y_hat()- self.y)**2)
    def gradient(self):
        return 2/self.N * (self.Xbar.T @ (self.y_hat()- self.y))
    def momentum(self):
        self.v0 = self.v0 * self.gamma + self.alpha * self.gradient()
        self.theta = self.theta - self.v0
    def gradient_descent(self) :
        self.theta -= self.alpha * self.gradient()
        return self.theta
    def fit(self, test = 1000):
        for i in range(test) :
            self.momentum()
            if i % 9999 == 0:
                print(f'Loss:{self.cost_function()}')
    def predict(self,X_test):
        ones = np.ones((X_test.shape[0],1))
        # return self.reverse_presict(np.concatenate((X_test, ones),axis=1) @ self.theta)
        return np.concatenate((X_test, ones),axis=1) @ self.theta

        # self.loss = loss

model = MyLinearRegression(X_train_new,y_train_new)
model.fit(10000)
y_pred = model.predict(X_test_new).reshape(-1)
print(f'My prediction:{reversed(y_pred)}')

model2 = LinearRegression()
model2.fit(X_train_new,y_train_new)
print(f'sklearn prediction:{reversed(model2.predict(X_test_new).reshape(-1))}')

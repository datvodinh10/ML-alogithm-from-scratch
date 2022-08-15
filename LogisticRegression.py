import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
X,y = datasets.load_breast_cancer(return_X_y=True)

X_train,X_test = X[:400],X[400:]
y_train,y_test = y[:400],y[400:]
X_test_new = (X_test-X_train.min())/(X_train.max()-X_train.min())
X_train_new = (X_train-X_train.min())/(X_train.max()-X_train.min())
y_test_new =  (y_test - y_train.min()) / (y_train.max()-y_train.min())
y_train_new  = (y_train - y_train.min()) / (y_train.max()-y_train.min())
class MyLogisticRegression():
    def __init__(self,X,y,lr=0.1):
        self.X = X
        self.y = y.reshape(-1,1)
        self.N = self.X.shape[0]
        self.ones = np.ones((self.N, 1))
        self.lr = lr
        self.Xbar = np.concatenate((self.X, self.ones), axis= 1)
        self.theta = np.random.randn((self.X.shape[1]+1), 1)
    @property
    def y_hat(self):
        return self.Xbar @ self.theta
    @property
    def sigmoid(self):
        return (1 / (1+ np.exp(-self.y_hat))).reshape(-1,1)
    @property
    def cost_function(self):
        return -np.mean((self.y * np.log(self.sigmoid)) + (1-self.y) * (np.log(1-self.sigmoid)))
    @property
    def gradient(self):
        return 1/ self.N *  np.dot(self.Xbar.T ,(self.sigmoid-self.y))
    def update_theta(self):
        self.theta -= self.lr * self.gradient
    
    def fit(self, test = 1000):
        for i in range(test+1) :
            self.update_theta()
            if i % 9999 == 0:
                print(f'Loss:{self.cost_function}')

    def predict(self,X_test):
        ones = np.ones((X_test.shape[0],1))
        matrix = 1 / (1+ np.exp(-(np.concatenate((X_test, ones),axis=1) @ self.theta)))
        return np.array([0 if i<0.5 else 1 for i in matrix]).reshape(matrix.shape)
        


model = MyLogisticRegression(X_train_new,y_train)
model.fit(10000)
a = model.predict(X_test_new).reshape(-1)-y_test
print(f'My accuracy:{(a.shape[0]-sum(abs(a)))/a.shape[0]}')

model2 = LogisticRegression()
model2.fit(X_train_new,y_train)
y_pred = model2.predict(X_test_new)
from sklearn.metrics import accuracy_score
print(f'Sklearn accuracy:{accuracy_score(y_pred,y_test)}')

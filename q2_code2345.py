import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("./hw3Data/emails.csv")
x = data.iloc[:, 1:3001].values
y = data.Prediction.values

# knn=KNeighborsClassifier(n_neighbors=1)
# kf=KFold(n_splits=5)
# for i, (train_index, test_index) in enumerate(kf.split(x)):
    
#     xTrain, xTest = x[train_index], x[test_index]
#     yTrain, yTest = y[train_index], y[test_index]

#     knn.fit(xTrain,yTrain)
#     yPred=knn.predict(xTest)
#     print("Fold " + str(i))
#     print("Accuracy " + str(accuracy_score(yTest,yPred)))
#     print("Precision " + str(precision_score(yTest,yPred)))
#     print("Recall " + str(recall_score(yTest,yPred)))

def sigma(z):
    return (1/(1+np.exp(-z))).astype(np.float64)
    
def crossEntropyLoss(yTrue, y_prob):
    return -np.mean((yTrue*np.log(y_prob+1e-12)+(1-yTrue)*np.log(1-y_prob+1e-12)))

class logRegModel:
    def __init__(self, maxIterations=100, learning_rate=0.01):
        self.maxIterations = maxIterations
        self.learning_rate = learning_rate

    def trainModel(self, xTrain, yTrain):
        self._n_features, self._n_examples = xTrain.shape[1], xTrain.shape[0]

        theta = np.random.uniform(low=-0.01, high=0.01, size=self._n_features+1).astype(np.float64)

        self.best_theta = theta
        loss = np.inf
        best_loss = np.inf
        for i in range(self.maxIterations):

            arr = np.ones((self._n_examples, 1), dtype=np.float64)
            xTrainA = np.hstack((arr, xTrain.astype(np.float64)))
            yHat = sigma(np.dot(xTrainA, theta))

            grad = 1/self._n_examples * np.dot(xTrainA.T, yHat-yTrain)
            theta -= self.learning_rate*grad

            loss = crossEntropyLoss(yTrain, yHat)

            if loss < best_loss:
                best_loss = loss
                self.best_theta = theta

    def predict(self,xTest): 
        arr = np.ones((xTest.shape[0], 1), dtype=np.float64)
        xTest_app = np.hstack((arr, xTest.astype(np.float64)))
        ySigma=sigma(np.dot(xTest_app, self.best_theta)).tolist()
        yPred=np.array([0 if k < 0.5 else 1 for k in ySigma])

        return yPred


# kf=KFold(n_splits=5)
# for i, (train_index, test_index) in enumerate(kf.split(x)):
    
#     xTrain, xTest = x[train_index], x[test_index]
#     yTrain, yTest = y[train_index], y[test_index]
#     model = logRegModel(1000,0.0005)
#     model.trainModel(xTrain,yTrain)
#     yPred=model.predict(xTest)
#     print("Fold " + str(i))
#     print("Accuracy " + str(accuracy_score(yTest,yPred)))
#     print("Precision " + str(precision_score(yTest,yPred)))
#     print("Recall " + str(recall_score(yTest,yPred)))

# kArr=[1,3,5,7,10]
# avgArr=[]
# for k in kArr:
#     knn=KNeighborsClassifier(n_neighbors=k)
#     kf=KFold(n_splits=5)
#     curAccuracy=[]
#     for i, (train_index, test_index) in enumerate(kf.split(x)):
#         xTrain, xTest = x[train_index], x[test_index]
#         yTrain, yTest = y[train_index], y[test_index]

#         knn.fit(xTrain,yTrain)
#         yPred=knn.predict(xTest)
#         curAccuracy.append(accuracy_score(yTest,yPred))

#     print("At K: " + str(k) + " avg accuracy= " +  str(np.mean(curAccuracy)))
#     avgArr.append(np.mean(curAccuracy))

# plt.plot(kArr,avgArr)
# plt.xlabel('k')
# plt.ylabel('Average Accuracy')
# plt.title('kNN 5-Fold Cross Validation')
# plt.show()

xTrain,xTest=x[:4000],x[4000:]
yTrain,yTest=y[:4000],y[4000:]

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(xTrain,yTrain)
yPredict=knn.predict(xTest)
yProb=knn.predict_proba(xTest)

logreg=LogisticRegression(max_iter=20000)
logreg.fit(xTrain,yTrain)
yPredLR=logreg.predict(xTest)
yProbLR=logreg.predict_proba(xTest)

def ROC(yTrue, yScore):
    sortedIndex = np.argsort(yScore)[::-1]
    
    num_neg = (yTrue[sortedIndex] == 0).sum()
    num_pos = (yTrue[sortedIndex] == 1).sum()
    
    TP,FP,TPL = 0,0,0
    fprCoordinates,tprCoordinates = [0],[0]
    # Loop over the test set instances
    for i in range(len(yScore)):
        # Find thresholds where there is a pos instance on high side, neg instance on low side
        if i > 0 and yScore[sortedIndex[i]] != yScore[sortedIndex[i-1]] and yTrue[sortedIndex][i] == 0 and TP > TPL:
            FPR = FP / num_neg
            TPR = TP / num_pos
            fprCoordinates.append(FPR)
            tprCoordinates.append(TPR)
            TPL = TP
        if yTrue[sortedIndex][i] == 1:
            TP += 1
        else:
            FP += 1
    
    # Add the last point to the ROC curve
    FPR = FP / num_neg
    TPR = TP / num_pos
    fprCoordinates.append(FPR)
    tprCoordinates.append(TPR)

    return fprCoordinates, tprCoordinates

knnFPR,knnTPR=ROC(yTest,yProb[:,1]) #sending the probabilities of class-1(Spam prediction)
LRFPR,LRTPR=ROC(yTest,yProbLR[:,1])

plt.plot(knnFPR, knnTPR, label='kNN',color='blue')
plt.plot(LRFPR,LRTPR,label='Logistic Regression',color='orange')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold 
from sklearn import metrics
import matplotlib.pyplot as plt


filename = 'C:/Users/kosta/OneDrive/Υπολογιστής/New folder/ODEP/telco_2023.csv'

data = pd.read_csv(filename)
classlabel = data.iloc[:, -1]
attr = data.iloc[:, 0:-1]

k=5
kf = KFold(n_splits=k, random_state=None)
model = KNeighborsClassifier(n_neighbors=5)

acc_score = []
rec_score = []
pre_score = []

for train_index , test_index in kf.split(attr):
    X_train , X_test = attr.iloc[train_index,:],attr.iloc[test_index,:]
    y_train , y_test = classlabel[train_index] , classlabel[test_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
     
    acc = metrics.accuracy_score(pred_values , y_test)
    acc_score.append(acc)

    rec = metrics.recall_score(pred_values, y_test)
    rec_score.append(rec)

    pre = metrics.precision_score(pred_values , y_test)
    pre_score.append(pre)
     
avg_acc_score = sum(acc_score)/k
avg_rec_score = sum(rec_score)/k
avg_pre_score = sum(pre_score)/k
 
print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))
print('recall of each fold - {}'.format(rec_score))
print('Avg recall : {}'.format(avg_rec_score))
print('precision of each fold - {}'.format(pre_score))
print('Avg precision : {}'.format(avg_pre_score))
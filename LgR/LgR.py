import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics

#Importing the feature set
feature_log = pd.read_csv('feature_data_frame_log.csv')

#split dataset in features and target variable
X = feature_log.iloc[:, : 7] # Features
Y = feature_log.Label # Target variable

# split X and Y into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=16)

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, Y_train)

#Predicting for testing set
y_pred = logreg.predict(X_test)

# Model Evaluation - 

# Confusion Matrix
cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)
cnf_matrix
#Plotting the Confusion Matrix
class_names=[0,1] # name  of classes
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, fontsize = 15)
plt.yticks(tick_marks, class_names, fontsize = 15)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, annot_kws={'size': 30}, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label', fontsize = 20)
plt.xlabel('Predicted label', fontsize = 20)
plt.show()

#Other Metrics
target_names = ['open', 'close']
print(classification_report(Y_test, y_pred, target_names=target_names))

# ROC-AUC Curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr,tpr,label="AUC="+str(auc) )
plt.xlabel('FPR', fontsize = 20)
plt.ylabel('TPR', fontsize = 20)
plt.legend(loc=4, prop = { "size": 20 })
plt.show()

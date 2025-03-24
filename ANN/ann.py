import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

#Importing the data set
feature = pd.read_csv('feature_data_frame_log.csv')

#split dataset in features and target variable
X = feature.iloc[:, : 7] # Features
Y = feature.Label # Target variable

# split X and Y into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=16)


######### Instantiating the model  ############# 
model_feature = Sequential()

#Input Layer
model_feature.add(Input(shape = (7,)))

# Four Hidden Layer
model_feature.add(Dense(8, activation = 'relu'))
model_feature.add(Dropout(0.1))

model_feature.add(Dense(16, activation = 'relu'))
model_feature.add(Dropout(0.1))

model_feature.add(Dense(16, activation = 'relu'))
model_feature.add(Dropout(0.1))

model_feature.add(Dense(8, activation = 'relu'))
model_feature.add(Dropout(0.1))

# Output Layer
model_feature.add(Dense(1, activation = 'sigmoid'))

#Configuring the model (Specifying Optimization Algorithm, Loss Function, Evaluation Metric)
model_feature.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Model Summary
model_feature.summary()

#################################################

#Model Fit
model_feature.fit(X_train, Y_train, epochs = 500, validation_data = (X_test, Y_test))

#Predicting for testing set
y_pred = model_feature.predict(X_test)

#From the probability obtained, deciding the class
Y_pred = []
for i, val in enumerate(y_pred):
  if(val < 0.5):
    Y_pred.append(0)
  else:
    Y_pred.append(1)
len(Y_pred)
Y_pred = pd.DataFrame(Y_pred, columns = ['Label'])
Y_pred.head()

# Different Evaluation Metrics

# Confusion Matrix
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
cnf_matrix

#Plotting the Confusion Matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, fontsize = 15)
plt.yticks(tick_marks, class_names, fontsize = 15)
# Plotting Heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, annot_kws={'size': 22}, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label', fontsize = 20)
plt.xlabel('Predicted label', fontsize = 20)
plt.show()


#Other Metrics
target_names = ['open', 'close']
print(classification_report(Y_test, Y_pred, target_names=target_names))

# ROC-AUC Curve
y_pred_proba = model_feature.predict(X_test)
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr,tpr,label="AUC="+str(auc) )
plt.xlabel('FPR', fontsize = 20)
plt.ylabel('TPR', fontsize = 20)
plt.legend(loc=4, prop = { "size": 20 })
plt.show()

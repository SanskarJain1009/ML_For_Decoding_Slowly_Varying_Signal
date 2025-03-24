import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization,Dropout
import matplotlib.pyplot as plt


#Importing the training set for O1 
train_o1 = keras.utils.image_dataset_from_directory(
    directory = 'train_o1',
    labels="inferred",
    label_mode="int",
    batch_size=32)

#Importing the testing set for O1
test_o1 = keras.utils.image_dataset_from_directory(
    directory = 'test_o1',
    labels="inferred",
    label_mode="int",
    batch_size=32)


############## Normalization ##################
def normal(image, label):
	image = tf.cast(image/255, tf.float32)
	return image, label

train_o1 = train_o1.map(normal)
test_o1 = test_o1.map(normal)


##############  Create a CNN Model   ####################

model_o1 = Sequential()

model_o1.add(Conv2D(192, kernel_size = (3,3), padding = 'valid', activation = 'relu', input_shape = (256, 256,3)))
model_o1.add(BatchNormalization())
model_o1.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'valid'))

model_o1.add(Conv2D(128, kernel_size = (3,3), padding = 'valid', activation = 'relu', input_shape = (256, 256,3)))
model_o1.add(BatchNormalization())
model_o1.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'valid'))

model_o1.add(Conv2D(64, kernel_size = (3,3), padding = 'valid', activation = 'relu', input_shape = (256, 256,3)))
model_o1.add(BatchNormalization())
model_o1.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'valid'))

model_o1.add(Conv2D(32, kernel_size = (3,3), padding = 'valid', activation = 'relu', input_shape = (256, 256,3)))
model_o1.add(BatchNormalization())
model_o1.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'valid'))

model_o1.add(Flatten())

model_o1.add(Dense(32, activation = 'relu'))
model_o1.add(Dropout(0.1))

model_o1.add(Dense(16, activation = 'relu'))
model_o1.add(Dropout(0.1))

model_o1.add(Dense(8, activation = 'relu'))
model_o1.add(Dropout(0.1))

model_o1.add(Dense(1, activation = 'sigmoid'))

#Configuring the model (Specifying Optimization Algorithm, Loss Function, Evaluation Metric)
model_o1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Model Summary
model_o1.summary()

###########################################

#Model Fit
model_o1.fit(train_o1, epochs = 10, validation_data = test_o1)

#Training and Testing Accuracy Plot
plt.plot(model_oz.history.history['accuracy'], color = 'red', label = 'train')
plt.plot(model_oz.history.history['val_accuracy'], color = 'blue', label = 'test')
plt.legend()
plt.show()

#Loss Plot
plt.plot(model_oz.history.history['loss'], color = 'red', label = 'train')
plt.plot(model_oz.history.history['val_loss'], color = 'blue', label = 'test')
plt.legend()
plt.show()

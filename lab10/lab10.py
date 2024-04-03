#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import time
import os
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ### 1.Klasyfikacja obrazów

# In[ ]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() 
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# In[ ]:


X_train = X_train / 255
X_test = X_test / 255


# In[ ]:


plt.imshow(X_train[142], cmap="binary") 
plt.axis('off')
plt.show()


# In[ ]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
               "sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[142]]


# In[ ]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))


# In[ ]:


model.summary()
tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)


# In[ ]:


model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='sgd',
              metrics=['accuracy'])


# In[ ]:


def get_run_logdir(root_logdir): 
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") 
    return os.path.join(root_logdir, run_id)


# In[ ]:


root_logdir = os.path.join(os.curdir, "image_logs")
run_logdir = get_run_logdir(root_logdir)


# In[ ]:


tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)


# In[ ]:


validation_split = 0.1


# In[ ]:


history = model.fit(X_train, y_train, epochs=20,
                    validation_split=validation_split,
                    callbacks=[tensorboard_cb])


# In[ ]:


image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()


# In[ ]:


model.save('fashion_clf.h5')


# ### 2.Regresja

# #### Dane

# In[ ]:


housing = fetch_california_housing()


# In[ ]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(housing.data, housing.target, test_size=0.2)
X_train2, X_valid2, y_train2, y_valid2 = train_test_split(X_train2, y_train2, test_size=0.2)


# In[ ]:


scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train2)
X_valid2 = scaler.transform(X_valid2)
X_test2 = scaler.transform(X_test2)


# #### Model 1

# In[ ]:


model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.Dense(30, activation='relu', input_shape=(X_train2.shape[1],)))
model2.add(tf.keras.layers.Dense(1))


# In[ ]:


model2.compile(loss='mean_squared_error', optimizer='sgd')


# In[ ]:


early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01, verbose=1)


# In[ ]:


root_logdir = os.path.join(os.curdir, "housing_logs")
run_logdir = get_run_logdir(root_logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(run_logdir)


# In[ ]:


history = model2.fit(X_train2,
                    y_train2,
                    validation_data=(X_valid2, y_valid2),
                    epochs=100,
                    callbacks=[early_stopping, tensorboard_callback])


# In[ ]:


model2.save('reg_housing_1.h5')


# In[ ]:


mse = model2.evaluate(X_test2, y_test2)
print("MSE on test set:", mse)


# #### Model 2

# In[ ]:


model3 = tf.keras.models.Sequential()
model3.add(tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train2.shape[1],)))
model3.add(tf.keras.layers.Dense(100, activation='relu'))
model3.add(tf.keras.layers.Dense(1))


# In[ ]:


model3.compile(loss='mean_squared_error', optimizer='sgd')


# In[ ]:


root_logdir = os.path.join(os.curdir, "housing_logs")
run_logdir = get_run_logdir(root_logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(run_logdir)


# In[ ]:


history = model3.fit(X_train2,
                    y_train2,
                    validation_data=(X_valid2, y_valid2),
                    epochs=100,
                    callbacks=[early_stopping, tensorboard_callback])


# In[ ]:


model3.save('reg_housing_2.h5')


# #### Model 3

# In[ ]:


model4 = tf.keras.models.Sequential()
model4.add(tf.keras.layers.Dense(20, activation='relu', input_shape=(X_train2.shape[1],)))
model4.add(tf.keras.layers.Dense(20, activation='relu'))
model4.add(tf.keras.layers.Dense(50, activation='relu'))
model4.add(tf.keras.layers.Dense(50, activation='relu'))
model4.add(tf.keras.layers.Dense(1))


# In[ ]:


model4.compile(loss='mean_squared_error', optimizer='sgd')


# In[ ]:


root_logdir = os.path.join(os.curdir, "housing_logs")
run_logdir = get_run_logdir(root_logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(run_logdir)


# In[ ]:


history = model4.fit(X_train2,
                    y_train2,
                    validation_data=(X_valid2, y_valid2),
                    epochs=100,
                    callbacks=[early_stopping, tensorboard_callback])


# In[ ]:


model4.save('reg_housing_3.h5')


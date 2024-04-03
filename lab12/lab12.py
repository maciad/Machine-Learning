#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


# In[ ]:


dataset_name = 'tf_flowers'
[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
    dataset_name,
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"], as_supervised=True,
    with_info=True)


# In[ ]:


class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples


# In[ ]:


plt.figure(figsize=(12, 8))
index = 0
sample_images = train_set_raw.take(9) 
for image, label in sample_images:
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label])) 
    plt.axis("off")
    
plt.show(block=False)


# In[ ]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224]) 
    return resized_image, label


# In[ ]:


batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[ ]:


plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
print(sample_batch)

for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index]/255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")

plt.show()


# ### Budowa sieci

# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Rescaling(scale=1./255))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding='same',
                                  activation='relu', input_shape=[224, 224, 3]))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=5, padding='same',
                                  activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=5, padding='same',
                                  activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                  activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                  activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5)) 
model.add(tf.keras.layers.Dense(units=5, activation='softmax'))


# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Rescaling(scale=1./255))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding='same',
                                  activation='relu', input_shape=[224, 224, 3]))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same',
                                  activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                                  activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5)) 
model.add(tf.keras.layers.Dense(units=5, activation='softmax'))


# In[ ]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# In[ ]:


model.fit(train_set, epochs=10, validation_data=valid_set)


# In[ ]:


acc_train = model.evaluate(train_set)[1]
acc_valid = model.evaluate(valid_set)[1]
acc_test = model.evaluate(test_set)[1]
results = (acc_train, acc_valid, acc_test)


# In[ ]:


with open('simple_cnn_acc.pkl', 'wb') as f:
    pickle.dump(results, f)


# ### 2.3 Uczenie transferowe

# In[ ]:


def preprocess_x(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


# In[ ]:


batch_size = 32
train_set_x = train_set_raw.map(preprocess_x).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set_x = valid_set_raw.map(preprocess_x).batch(batch_size).prefetch(1)
test_set_x = test_set_raw.map(preprocess_x).batch(batch_size).prefetch(1)


# In[ ]:


plt.figure(figsize=(8, 8))
sample_batch = train_set_x.take(1)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index] / 2 + 0.5)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")
        
plt.show()


# In[ ]:


base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)


# In[ ]:


inputs = tf.keras.Input(shape=[224, 224, 3])
base = base_model(inputs, training=False)
avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(base)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(avg_pooling)
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)


# In[ ]:


for layer in base_model.layers:
    layer.trainable = False


# In[ ]:


model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# In[ ]:


model2.fit(train_set_x, epochs=6, validation_data=valid_set_x)


# In[ ]:


# for layer in base_model.layers:
#     layer.trainable = True


# In[ ]:


# model2.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#               metrics=['accuracy'])


# In[ ]:


# model2.fit(train_set_x, epochs=3, validation_data=valid_set_x)


# In[ ]:


acc_train2 = model2.evaluate(train_set_x)[1]
acc_valid2 = model2.evaluate(valid_set_x)[1]
acc_test2 = model2.evaluate(test_set_x)[1]
results2 = (acc_train2, acc_valid2, acc_test2)


# In[ ]:


with open('xception_acc.pkl', 'wb') as f:
    pickle.dump(results2, f)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from scipy.stats import reciprocal
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# In[ ]:


param_distribs = {
    'model__n_hidden': [0, 1, 2, 3],
    'model__n_neurons': np.arange(1,100),
    'model__learning_rate': reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
    'model__optimizer': ['adam', 'sgd', 'nesterov', 'momentum']
}


# In[ ]:


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, optimizer='sgd'):
    optimizer_map = {
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate),
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'nesterov': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True),
        'momentum': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
    }

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=housing.data.shape[1]))
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    
    optimizer = optimizer_map.get(optimizer)

    model.compile(loss='mse', optimizer=optimizer)
    return model


# In[ ]:


es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)


# In[ ]:


keras_reg = KerasRegressor(build_model)

keras_reg.fit(X_train, y_train, epochs=100,
              validation_data = (X_valid, y_valid),
              callbacks = [es])


# In[ ]:


rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs,
                                   n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(X=X_train, y=y_train, epochs=100,
                 validation_data=(X_valid, y_valid),
                 verbose=0)


# In[ ]:


best_params = rnd_search_cv.best_params_


# In[ ]:


with open('rnd_search_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)
    
with open('rnd_search_scikeras.pkl', 'wb') as f:
    pickle.dump(rnd_search_cv, f)


# In[ ]:


def build_model_kt(hp):
    n_hidden = hp.Int('n_hidden', min_value=0, max_value=3)
    n_neurons = hp.Int('n_neurons', min_value=1, max_value=100)
    learning_rate = hp.Float('learning_rate', min_value=3e-4, max_value=3e-2, sampling='log')
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'nesterov', 'momentum'])
    
    optimizer_map = {
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate),
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'nesterov': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True),
        'momentum': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
    }
    optimizer = optimizer_map.get(optimizer)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    return model


# In[ ]:


random_search_tuner = kt.RandomSearch(
    build_model_kt, objective='val_mse', max_trials=10, overwrite=True,
    directory="my_california_housing", project_name="my_rnd_search")


# In[ ]:


root_logdir = os.path.join(random_search_tuner.project_dir, 'tensorboard')
tb = tf.keras.callbacks.TensorBoard(root_logdir)


# In[ ]:


random_search_tuner.search(X_train, y_train, epochs=100,
                           validation_data=(X_valid, y_valid),
                           callbacks=[tb, es])


# In[ ]:


best_params_kt = random_search_tuner.get_best_hyperparameters(num_trials=1)[0]
best_params = best_params_kt.values

best_model = random_search_tuner.get_best_models(num_models=1)[0]


# In[ ]:


with open('kt_search_params.pkl', 'wb') as f:
    pickle.dump(best_params_kt, f)    


# In[ ]:


best_model.fit(X_train, y_train, epochs=100,
              validation_data = (X_valid, y_valid),
              callbacks = [es])
best_model.save('kt_best_model.h5')


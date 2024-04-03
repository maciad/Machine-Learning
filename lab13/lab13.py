#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import pandas as pd
import tensorflow as tf


# ### 2.1 Pobieranie danych

# In[ ]:


tf.keras.utils.get_file(
"bike_sharing_dataset.zip",
"https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
cache_dir=".",
extract=True
)


# ### 2.2 Przygotowanie danych

# In[ ]:


df = pd.read_csv('datasets/hour.csv',
                 parse_dates={'datetime': ['dteday', 'hr']},
                 date_format='%Y-%m-%d %H',
                 index_col='datetime')


# In[ ]:


df = df[['casual', 'registered', 'cnt', 'temp', 'atemp', 'hum', 'windspeed', 'holiday', 'weekday', 'workingday', 'weathersit']]


# In[ ]:


df = df.asfreq('h')


# In[ ]:


df[['casual', 'registered', 'cnt']] = df[['casual', 'registered', 'cnt']].fillna(0)
df[['temp', 'atemp', 'hum', 'windspeed']] = df[['temp', 'atemp', 'hum', 'windspeed']].interpolate()
df[['holiday', 'weekday', 'workingday', 'weathersit']] = df[['holiday', 'weekday', 'workingday', 'weathersit']].ffill()


# In[ ]:


df.notna().sum()


# In[ ]:


df[['casual', 'registered', 'cnt', 'weathersit']].describe()


# In[ ]:


df.casual /= 1e3
df.registered /= 1e3
df.cnt /= 1e3
df.weathersit /= 4


# In[ ]:


df_2weeks = df[:24 * 7 * 2]
df_2weeks[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))


# In[ ]:


df_daily = df.resample('W').mean()
df_daily[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))


# ### 2.3 Wskaźniki bazowe

# In[ ]:


df_sh_d = df['cnt'].shift(24)
df_sh_w = df['cnt'].shift(24*7)


# In[ ]:


mae_daily = (df['cnt'] - df_sh_d).abs().mean() * 1e3
mae_weekly = (df['cnt'] - df_sh_w).abs().mean() * 1e3


# In[ ]:


with open('mae_baseline.pkl', 'wb') as f:
    pickle.dump((mae_daily, mae_weekly), f)


# ### 2.4 Predykcja przy pomocy sieci gęstej

# In[ ]:


cnt_train = df['cnt']['2011-01-01 00:00':'2012-06-30 23:00']
cnt_valid = df['cnt']['2012-07-01 00:00':]


# In[ ]:


seq_len = 1 * 24

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_train.to_numpy(),
    targets=cnt_train[seq_len:],
    sequence_length=seq_len,
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_valid.to_numpy(),
    targets=cnt_valid[seq_len:],
    sequence_length=seq_len,
    batch_size=32
)


# In[ ]:


model = tf.keras.Sequential([
tf.keras.layers.Dense(1, input_shape=[seq_len])
])


# In[ ]:


optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
loss = tf.keras.losses.Huber(delta=1.0)


# In[ ]:


model.compile(optimizer=optimizer,
              loss=loss, 
              metrics=['mae']
)


# In[ ]:


history = model.fit(train_ds,
                    validation_data=valid_ds,
                    epochs=20,
                    batch_size=32)


# In[ ]:


val_loss, val_mae = model.evaluate(valid_ds)


# In[ ]:


with open('mae_linear.pkl', 'wb') as f:
    pickle.dump((val_mae,), f)
    
model.save('model_linear.h5')


# ### 2.5 Prosta sieć rekurencyjna

# In[ ]:


model2 = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])
])


# In[ ]:


optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.005)


# In[ ]:


model2.compile(optimizer=optimizer2,
              loss='mean_squared_error',
              metrics=['mae']
)


# In[ ]:


history2 = model2.fit(train_ds,
                      validation_data=valid_ds,
                      epochs=20,
                      batch_size=32)


# In[ ]:


val_loss2, val_mae2 = model2.evaluate(valid_ds)


# In[ ]:


with open('mae_rnn1.pkl', 'wb') as f:
    pickle.dump((val_mae2,), f)
    
model2.save('model_rnn1.h5')


# In[ ]:


model3 = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 1]),
    tf.keras.layers.Dense(1)
])


# In[ ]:


model3.compile(optimizer='adam',
               loss='mean_squared_error',
               metrics=['mae']
)


# In[ ]:


history3 = model3.fit(train_ds,
                      validation_data=valid_ds,
                      epochs=20,
                      batch_size=32)


# In[ ]:


val_loss3, val_mae3 = model3.evaluate(valid_ds)


# In[ ]:


with open('mae_rnn32.pkl', 'wb') as f:
    pickle.dump((val_mae3,), f)
    
model3.save('model_rnn32.h5')


# ### 2.6 Głęboka RNN

# In[ ]:


model4 = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 1], return_sequences=True),
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 1], return_sequences=True),
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 1]),
    tf.keras.layers.Dense(1)
])


# In[ ]:


model4.compile(optimizer='adam',
               loss='mean_squared_error',
               metrics=['mae']
)


# In[ ]:


history4 = model4.fit(train_ds,
                      validation_data=valid_ds,
                      epochs=20,
                      batch_size=32)


# In[ ]:


val_loss4, val_mae4 = model4.evaluate(valid_ds)


# In[ ]:


with open('mae_rnn_deep.pkl', 'wb') as f:
    pickle.dump((val_mae4,), f)
    
model4.save('model_rnn_deep.h5')


# ### 2.7 Model wielowymiarowy

# In[ ]:


cnt_train2 = df[['cnt', 'weathersit', 'atemp', 'workingday']]['2011-01-01 00:00':'2012-06-30 23:00']
cnt_valid2 = df[['cnt', 'weathersit', 'atemp', 'workingday']]['2012-07-01 00:00':]


# In[ ]:


seq_len = 1 * 24

train_ds2 = tf.keras.utils.timeseries_dataset_from_array(
    cnt_train2.to_numpy(),
    targets=cnt_train2['cnt'][seq_len:],
    sequence_length=seq_len,
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_ds2 = tf.keras.utils.timeseries_dataset_from_array(
    cnt_valid2.to_numpy(),
    targets=cnt_valid2['cnt'][seq_len:],
    sequence_length=seq_len,
    batch_size=32
)


# In[ ]:


model5 = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 4]),
    tf.keras.layers.Dense(1)
])


# In[ ]:


model5.compile(optimizer='adam',
               loss='mean_squared_error',
               metrics=['mae']
)


# In[ ]:


history5 = model5.fit(train_ds2,
                      validation_data=valid_ds2,
                      epochs=20,
                      batch_size=32)


# In[ ]:


val_loss5, val_mae5 = model5.evaluate(valid_ds2)


# In[ ]:


with open('mae_rnn_mv.pkl', 'wb') as f:
    pickle.dump((val_mae5,), f)
    
model5.save('model_rnn_mv.h5')


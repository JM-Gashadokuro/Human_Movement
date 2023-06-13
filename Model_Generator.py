import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from keras.utils.np_utils import to_categorical  # one-hot-encoding
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# load data into memory
df_train = pd.read_csv('input/train.csv')
df_train['Activity'] = pd.Categorical(df_train['Activity']).codes

x_train = df_train.drop(columns=['subject', 'Activity'], axis='columns')
y_train = df_train['Activity']

df_test = pd.read_csv('input/test.csv')
df_test['Activity'] = pd.Categorical(df_test['Activity']).codes

features_test = df_test.drop(columns=['subject', 'Activity'], axis='columns')
target_test = df_test['Activity']

x_test, x_validate, y_test, y_validate = train_test_split(features_test, target_test, test_size=0.5, random_state=2)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_validate=to_categorical(y_validate)

# input is a 561-dimensional vector
input_shape = (561,)
num_classes = 6

model = Sequential()
model.add(Input(shape=input_shape))
model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

opt = Adam(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
mcp_output = ModelCheckpoint('output/mdl_best_output.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

epochs = 50
batch_size = 100
cnn_history = model.fit(x_train, y_train,
                        epochs=epochs, validation_data=(x_validate, y_validate),
                        verbose=1, steps_per_epoch=x_train.shape[0] // batch_size,
                        callbacks=[lr_reduction, early_stop, mcp_output])

model.save("output/mdl_output.h5")

model1 = keras.models.load_model("output/mdl_best_output.hdf5", compile=False)
model1.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

loss, accuracy = model1.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = model1.evaluate(x_validate, y_validate, verbose=1)

print("Best model validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Best model test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)

print("Final model validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Final model test: accuracy = %f  ;  loss = %f" % (accuracy, loss))


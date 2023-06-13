import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from keras.utils.np_utils import to_categorical  # one-hot-encoding
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# input is a 561-dimensional vector
input_shape = (561,)
num_classes = 6

model = Sequential()
model.add(Input(shape=(561,)))
model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))
model.summary()

opt = Adam(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

# load data into memory
df_train = pd.read_csv('input/train.csv')
df_train['Activity'] = pd.Categorical(df_train['Activity']).codes
df_train.head()
x_train = df_train.drop(columns=['subject', 'Activity'], axis='columns')
y_train = df_train['Activity']

x_train=np.asarray(x_train)
y_train=np.asarray(y_train)

df_test = pd.read_csv('input/test.csv')
df_test['Activity'] = pd.Categorical(df_test['Activity']).codes
df_test.head()
features_test = df_test.drop(columns=['subject', 'Activity'], axis='columns')
target_test = df_test['Activity']

features_test=np.asarray(features_test)
target_test=np.asarray(target_test)

x_test, x_validate, y_test, y_validate = train_test_split(features_test, target_test, test_size=0.5, random_state=2)

x_test=np.asarray(x_test)
y_test=np.asarray(y_test)

x_validate=np.asarray(x_validate)
y_validate =np.asarray(y_validate )

print(x_train[0].shape)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_validate=to_categorical(y_validate)
print(y_validate)

lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', restore_best_weights=True)
mcp_output = ModelCheckpoint('output/mdl_best_output.hdf5', save_best_only=True, monitor='val_loss', mode='min')

epochs = 20
batch_size = 50
cnn_history = model.fit(x_train, y_train,
                        epochs=epochs, validation_data=(x_validate, y_validate),
                        verbose=1, steps_per_epoch=x_train.shape[0] // batch_size,
                        callbacks=[lr_reduction, early_stop, mcp_output])

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
model.save("output/mdl_output.h5")

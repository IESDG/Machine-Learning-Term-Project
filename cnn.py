import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import Sequential

# First we create our tf Database using train and test files
# We use %20 of the training data to validate our training

# Training data
data_train = tf.keras.utils.image_dataset_from_directory(
    'train',
    image_size=(100, 100),
    validation_split=0.2,
    subset="training",
    seed=13)

# Validation data
data_val = tf.keras.utils.image_dataset_from_directory(
    'train',
    image_size=(100, 100),
    validation_split=0.2,
    subset="validation",
    seed=13)

# Testing data
data_test = tf.keras.utils.image_dataset_from_directory(
    'test',
    image_size=(100, 100),
    )

# Labels
class_names = data_train.class_names

# Rescaling our data from 0-255 to 0-1
normalization_layer = layers.Rescaling(1./255)

num_classes = len(class_names)

# Loading our model -> Cnn
# 3 layers using 'relu' activation
# %10 dropout chance to add variance
# And a dense layer using 128 nodes and a dense layer using number of labels as node counter
model = Sequential([
  layers.Rescaling(1./255, input_shape=(100, 100, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(num_classes)
])

# Compiling our model, using adam optimizer
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Number of epochs to retrain our model
epochs = 3

# Training our model
history = model.fit(
  data_train,
  validation_data=data_val,
  epochs=epochs
)

# Cross Validating our data
print("Testing")
testing = model.evaluate(data_test)
print("test loss, test acc: ", testing)

# A plot to give our model a visual representation
# https://www.tensorflow.org/tutorials/images/classification

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
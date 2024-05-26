import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create the model
model = models.Sequential()
# First convolutional and pooling layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
# Second convolutional and pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Third convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# Fourth convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Flatten layer
model.add(layers.Flatten())
# First hidden layer
model.add(layers.Dense(128, activation='relu'))
# Second hidden layer
model.add(layers.Dense(64, activation='relu'))
# Output layer
model.add(layers.Dense(10))

model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model (set epochs to 20)
history = model.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)


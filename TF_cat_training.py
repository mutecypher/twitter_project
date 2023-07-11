import tensorflow as tf
from tensorflow.keras import layers
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold


image_height = 256
image_width = 256
batch_size = 214
validation_split = 0.40
num_epochs = 12
learning_rate = 0.00005
seedy = random.randint(100, 999)


# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # Three output classes: cats, mouthy cats, not cats
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# Preprocess and load the image data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Now_train',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    validation_split=validation_split,
    subset='training'
   , seed = seedy
)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Now_train',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    validation_split=validation_split,
    subset='validation',
    seed = seedy
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    '/Volumes/Elements/GitHub/cats_with_birds/For_Training/test',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    seed = seedy
)


# Train the model
model.fit(
    train_data,
    validation_data=validation_data,
    epochs=num_epochs
)


# Save the trained model
model.save('/Volumes/Elements/GitHub/cats_with_birds/Torchy/model_tf')

loss, accuracy = model.evaluate(test_data)
print()
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')
print()
loss, accuracy = model.evaluate(validation_data)
print(f'validation_data loss: {loss:.4f}')
print(f'validation_data accuracy: {accuracy:.4f}')

print()
loss, accuracy = model.evaluate(train_data)
print(f'train_data loss: {loss:.4f}')
print(f'train_data accuracy: {accuracy:.4f}')
print()

# Load the saved model
model = tf.keras.models.load_model('/Volumes/Elements/GitHub/cats_with_birds/Torchy/model_tf')


# Extract the true labels from the dataset
test_labels = []
##for images, labels in test_dataset:
for images, labels in test_data:
    test_labels.extend(labels.numpy())

train_labels = []
for images, labels in train_data:
    train_labels.extend(labels.numpy())

validation_labels = []
for images, labels in validation_data:
    validation_labels.extend(labels.numpy())


predictions_train = model.predict(train_data)
predicted_labels_train = np.argmax(predictions_train, axis=1)

print(f'predicted_labels from test data are {predicted_labels_train}')

# Make predictions on the test dataset
##predictions = model.predict(test_dataset)
predictions_val = model.predict(validation_data)
predicted_labels_val = np.argmax(predictions_val, axis=1)

print(f'predicted_labels from test data are {predicted_labels_val}')


predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

print(f'predicted_labels form test data are {predicted_labels}')

# Compute the confusion matrix
cm_test = confusion_matrix(test_labels, predicted_labels)

# Compute the confusion matrix
cm_val = confusion_matrix(validation_labels, predicted_labels_val)

cm_train = confusion_matrix(train_labels, predicted_labels_train)
# Plot the confusion matrix as a heatmap
categories = ["cat", "cat in mouth", "not cat"]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix- train')
plt.savefig('/Volumes/Elements/GitHub/cats_with_birds/Torchy/Confusion_Matrix_train.png')

categories = ["cat", "cat in mouth", "not cat"]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix- Validation')
plt.savefig('/Volumes/Elements/GitHub/cats_with_birds/Torchy/Confusion_Matrix_val.png')


categories = ["cat", "cat in mouth", "not cat"]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix- test')
plt.savefig('/Volumes/Elements/GitHub/cats_with_birds/Torchy/Confusion_Matrix_test.png')



def plot_confusion_matrix_per(confusion_matrix, labels):
    # Calculate row-wise sums
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    # Calculate percentages
    confusion_matrix_percent = confusion_matrix / row_sums.astype(float) * 100

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_matrix_percent, annot=True, fmt=".1f", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.title("Confusion Matrix (Percentages)")
    plt.show()


labels = ["Cat", "Cat with Something in Mouth", "Not Cat"]

# Plot the confusion matrix with percentages
##plot_confusion_matrix_per(cm_train, labels)

##plot_confusion_matrix_per(cm_val, labels)

plot_confusion_matrix_per(cm_test, labels)
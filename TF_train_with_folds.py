import tensorflow as tf
from tensorflow.keras import layers
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

image_height = 256
image_width = 256
batch_size = 32
validation_split = 0.4
num_epochs = 40
learning_rate = 0.0001
seedy = random.randint(100, 999)

# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')  # Three output classes: cats, mouthy cats, not cats
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Preprocess and load the image data
image_data = tf.keras.preprocessing.image_dataset_from_directory(
    '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Now_train',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    validation_split=validation_split,
    subset = 'training',
    seed=seedy
)

# Split the data into training, validation, and test sets
train_data = image_data
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Now_train',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    validation_split=validation_split,
    subset='validation',
    seed=seedy
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Now_train',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    validation_split=0.5,
    subset = 'validation',
    shuffle=False,
    seed=seedy
)

# Train the model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=num_epochs
)

# Save the trained model
model.save('/Volumes/Elements/GitHub/cats_with_birds/Torchy/model_tf')

# Load the saved model
model = tf.keras.models.load_model('/Volumes/Elements/GitHub/cats_with_birds/Torchy/model_tf')

# Load the test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Now_train',
    image_size=(256, 256),
    batch_size=batch_size,
    shuffle=False
)

# Extract the true labels from the dataset
true_labels = []
for images, labels in test_dataset:
    true_labels.extend(labels.numpy())

# Make predictions on the test dataset
predictions = model.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix as a heatmap
categories = ["cat", "cat in mouth", "not a cat"]


# Plot the confusion matrix with counts
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

    # Plot the confusion matrix with percentages
row_sums = cm.sum(axis=1, keepdims=True)
cm_percent = cm / row_sums.astype(float) * 100
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Percentages)')
plt.savefig('/Volumes/Elements/GitHub/cats_with_birds/Torchy/Confusion_Matrix_folds.png')

fold += 1
    
def plot_confusion_matrix(confusion_matrix, labels):
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
    plt.savefig('/Volumes/Elements/GitHub/cats_with_birds/Torchy/Confusion_Matrix_folds_pct.png')


labels = ["Cat", "Cat with Something in Mouth", "Not Cat"]

# Plot the confusion matrix with percentages
plot_confusion_matrix(cm, labels)
import os
import tensorflow as tf

# Check if a saved model exists
saved_model_path = "path/to/saved/model"
if os.path.exists(saved_model_path):
    # Load the saved model
    model = tf.keras.models.load_model(saved_model_path)
    print("Loaded model from", saved_model_path)
else:
    # Build the model from scratch
    model = tf.keras.Sequential()
    # Add your model layers here
    # ...

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
num_epochs = 10
checkpoint_path = "path/to/save/checkpoints/model-{epoch:02d}.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False,
                                                         save_freq='epoch')
model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset, callbacks=[checkpoint_callback])

# Save the final trained model
final_model_path = "path/to/save/final/model"
model.save(final_model_path)
print("Saved final model to", final_model_path)

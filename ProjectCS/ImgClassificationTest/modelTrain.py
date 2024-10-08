import pandas as pd
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 

image_size = (200, 200)
EPOCHS = 50
MAX_IMAGES = 1000

def load_images_and_labels(image_folder, labels_df, max_images = 500):
    images = []
    labels = []

    for index, row in labels_df.iterrows():
        if len(images) > max_images:
            break

        img_path = os.path.join(image_folder, str(row['ID']) + '.png')

        img = cv2.imread(img_path)  # Load image with OpenCV
        img = cv2.resize(img, image_size)  # Resize to target size
        img = img / 255.0  # Normalize pixel values to [0, 1]

        images.append(img)
        labels.append(row['country'])

    return np.array(images), np.array(labels)           # return sample vector + label vector

image_folder = 'Data'
labels_df = pd.read_csv('coordinates_with_country.csv')

X, y = load_images_and_labels(image_folder, labels_df, MAX_IMAGES)

# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check shapes
print(f'Images shape: {X.shape}')
print(f'Labels shape: {y_encoded.shape}')


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True)


from tensorflow.keras import layers, models

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(len(np.unique(y_encoded)), activation='sigmoid')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy for integer labels
              metrics=['accuracy'])

model.summary()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the augmentation to your data
datagen.fit(X_train)

# Use the augmented data during training
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=EPOCHS, validation_data=(X_test, y_test))



test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')



import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# # model.save('image_classifier.h5')
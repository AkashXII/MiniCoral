import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
datagen=ImageDataGenerator(#rescales images and splits 20% for validation
    rescale=1./255,
    validation_split=0.2
)
train_data=datagen.flow_from_directory(#80% for validation
    "dataset/",
    target_size=(128,128),
    batch_size=8,
    class_mode='binary',
    subset='training'
)
val_data=datagen.flow_from_directory(#20% for validation
    "dataset/",
    target_size=(128,128),
    batch_size=8,
    class_mode='binary', #labels images as binary(0 or 1)
    subset='validation'
)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(train_data, validation_data=val_data, epochs=5)

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Training Progress")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

model.save("coral_model.h5")
print(" Model save check")

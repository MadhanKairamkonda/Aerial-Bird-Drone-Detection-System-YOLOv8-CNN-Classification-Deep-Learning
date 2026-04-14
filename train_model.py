# ================================
# IMPORTS
# ================================
import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# ================================
# STEP 2 & 3: PREPROCESSING + AUGMENTATION
# ================================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "data/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    "data/valid",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

test_data = val_gen.flow_from_directory(
    "data/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

print("Class mapping:", train_data.class_indices)

# ================================
# STEP 4: MODEL BUILDING
# ================================

# Custom CNN
cnn_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Transfer Learning
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

transfer_model = models.Model(inputs=base_model.input, outputs=output)

# ================================
# STEP 5: TRAINING
# ================================
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
transfer_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# Train CNN
start = time.time()
cnn_history = cnn_model.fit(train_data, validation_data=val_data, epochs=10, callbacks=callbacks)
cnn_time = time.time() - start

# Train Transfer Model
start = time.time()
transfer_history = transfer_model.fit(train_data, validation_data=val_data, epochs=10, callbacks=callbacks)
transfer_time = time.time() - start

# ================================
# STEP 6: EVALUATION
# ================================
cnn_loss, cnn_acc = cnn_model.evaluate(test_data)
transfer_loss, transfer_acc = transfer_model.evaluate(test_data)

print("\nCNN Accuracy:", cnn_acc)
print("Transfer Accuracy:", transfer_acc)

preds = transfer_model.predict(test_data)
pred_labels = (preds > 0.5).astype(int)

print("\nConfusion Matrix:\n", confusion_matrix(test_data.classes, pred_labels))
print("\nClassification Report:\n", classification_report(test_data.classes, pred_labels))

# ================================
# PLOT
# ================================
plt.plot(transfer_history.history['accuracy'], label='Train Accuracy')
plt.plot(transfer_history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.show()

# ================================
# STEP 7: COMPARISON
# ================================
print("\n--- MODEL COMPARISON ---")
print(f"CNN Accuracy: {cnn_acc:.4f}, Time: {cnn_time:.2f}s")
print(f"Transfer Accuracy: {transfer_acc:.4f}, Time: {transfer_time:.2f}s")

# Save best model
transfer_model.save("final_model.h5")

print("\n✅ Model saved as final_model.h5")
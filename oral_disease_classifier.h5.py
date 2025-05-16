# Step 1: Import Required Libraries
import os
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Add this line to enable eager execution
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns  # Add this import

# Step 2: Set Constants
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10  # Changed from 100 to 10
NUM_CLASSES = 6
DATASET_PATH = r"C:\SMESTER6\AI&ES\projectaies2\oral_diseases_dataset"

# Step 3: Prepare Data with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8,1.2],
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

class_names = list(train_generator.class_indices.keys())
print("Classes:", class_names)

# Step 4: Build Model with Transfer Learning
base_model = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# Freeze the base model
base_model.trainable = False

# Create the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Add learning rate scheduler
initial_learning_rate = 0.001
decay_steps = 2000
decay_rate = 0.95

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

# Use the learning rate schedule in the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model with the optimizer
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Add callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    min_delta=0.001
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Remove ReduceLROnPlateau callback
callbacks = [
    early_stopping,
    checkpoint
]

# Add class weights
class_weights = {
    0: 1.3,  # caries
    1: 1.3,  # cavity
    2: 1.2,  # discoloration
    3: 1.2,  # gingivitis
    4: 1.0,  # healthy
    5: 1.3   # ulcer
}

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# Save the final model
model.save("oral_disease_classifier.keras")  # Changed from .h5 to .keras

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Save training history
history_dict = history.history
with open('training_history.json', 'w') as f:
    json.dump(history_dict, f)

# Generate evaluation metrics
val_predictions = model.predict(val_generator)
val_pred_classes = np.argmax(val_predictions, axis=1)
true_classes = val_generator.classes

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

plot_confusion_matrix(true_classes, val_pred_classes, class_names)
print(classification_report(true_classes, val_pred_classes, target_names=class_names))

# Modify training parameters
EPOCHS = 10  # Changed from 50 to 10
BATCH_SIZE = 16  # Smaller batch size for better generalization

# Add learning rate scheduling
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

# Use a more sophisticated optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile with additional metrics
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Add class weights to handle imbalanced data
class_weights = {
    0: 1.0,  # cavity
    1: 1.0,  # healthy
    2: 1.2,  # discoloration
    3: 1.3,  # caries
    4: 1.2,  # gingivitis
    5: 1.3   # ulcer
}


# Add transfer learning with pre-trained model
base_model = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# Freeze the base model
base_model.trainable = False

# Add custom layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])


def validate_dataset(dataset_path):
    class_counts = {}
    min_required = 100  # minimum images per class
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = count
            if count < min_required:
                print(f"Warning: {class_name} has only {count} images. Recommend at least {min_required}")
    
    return class_counts

# Call before training
class_counts = validate_dataset(DATASET_PATH)
print("Dataset distribution:", class_counts)


# Create individual models for each disease
disease_models = {}
for disease in class_names:
    # Create and train a binary classifier for each disease
    binary_model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    
    # Create a new optimizer instance for each model
    binary_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compile the model with the new optimizer
    binary_model.compile(
        optimizer=binary_optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Prepare binary dataset
    binary_train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=[disease, 'healthy'],
        subset='training'
    )
    
    binary_val_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=[disease, 'healthy'],
        subset='validation'
    )
    
    # Train the model
    binary_model.fit(
        binary_train_generator,
        validation_data=binary_val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Save the model
    disease_models[disease] = binary_model

# Function to predict disease
def predict_disease(image_path):
    img = image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    results = {}
    for disease, model in disease_models.items():
        prediction = model.predict(img_array)
        results[disease] = prediction[0][0]
    
    # Get the disease with highest probability
    predicted_disease = max(results, key=results.get)
    confidence = results[predicted_disease]
    
    return predicted_disease, confidence, results


def get_class_weights(class_names):
    """Centralized class weight definition"""
    weights = {
        'cavity': 1.3,
        'healthy': 1.0,
        'discoloration': 1.2,
        'caries': 1.3,
        'gingivitis': 1.2,
        'ulcer': 1.3
    }
    return {i: weights[name] for i, name in enumerate(class_names)}

def build_model(num_classes):
    """Model architecture definition"""
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    """Main training pipeline"""
    # Data preparation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    class_names = list(train_generator.class_indices.keys())
    class_weights = get_class_weights(class_names)
    
    # Model setup
    model = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Training
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        class_weight=class_weights
    )
    
    return model, class_names

if __name__ == '__main__':
    model, class_names = train_model()
    model.save("models/oral_disease_classifier.keras")

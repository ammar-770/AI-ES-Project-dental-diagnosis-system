import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(image_size=224, num_classes=6):
    """
    Creates and returns the CNN model for oral disease classification
    
    Args:
        image_size (int): Size of input images (default: 224)
        num_classes (int): Number of disease classes to predict (default: 6)
        
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_class_names():
    """
    Returns the list of class names for oral disease classification
    
    Returns:
        list: List of class names
    """
    return ['caries', 'cavity', 'discoloration', 'gingivitis', 'healthy', 'ulcer']

def preprocess_image(img):
    """
    Preprocesses an image for model prediction
    
    Args:
        img: PIL Image object
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    img = tf.image.resize(img, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return tf.expand_dims(img_array, axis=0)
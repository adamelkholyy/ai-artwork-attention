import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

"""
Model wrapper class for resolving model loading errors.
The custom model wrapper is needed to load the model due to the inclusion of a custom bias layer, 
initiailised so as to eliminate the effects of a class imbalance in the data. 
The model was developed using tensorflow 2.15.1 however it is compatable with later versions 
(this code runs on 2.16.1) using the wrapper. Wrapper includes a function to process images 
before predictions, necessary because the model will only take RGB images of dimension 256x256.
"""
class model_wrapper:
    def __init__(self, model_path):
        try:
            self.model = load_model(model_path, compile=False, custom_objects={ "Dense": self.tf_wrapper})
            self.model.summary()
        except Exception as e:
            print(f"Failed to load model: {e}")

    @staticmethod
    def tf_wrapper(*args, **kwargs):
        config = kwargs.get('config', {})

        # custom bias layer: constant value = log(pos/neg) where
        # pos is number of ai images, neg is number of human-made
        if kwargs.get('name', None) == "dense_1":
            return tf.keras.layers.Dense(
                units=config.get('units', 1),
                activation=config.get('activation', 'sigmoid'),
                use_bias=config.get('use_bias', True),
                kernel_initializer=tf.keras.initializers.get(config.get('kernel_initializer', 'GlorotUniform')),
                bias_initializer=tf.keras.initializers.Constant(value=0.8498341056885719),
                kernel_regularizer=tf.keras.regularizers.get(config.get('kernel_regularizer')),
                bias_regularizer=tf.keras.regularizers.get(config.get('bias_regularizer')),
                activity_regularizer=tf.keras.regularizers.get(config.get('activity_regularizer')),
                kernel_constraint=tf.keras.constraints.get(config.get('kernel_constraint')),
                bias_constraint=tf.keras.constraints.get(config.get('bias_constraint'))
            )
        # regular dense layer 
        else:
            return tf.keras.layers.Dense(
                units=config.get('units', 128),
                activation=config.get('activation', 'relu'),
                use_bias=config.get('use_bias', True),
                kernel_initializer=tf.keras.initializers.get(config.get('kernel_initializer', 'GlorotUniform')),
                bias_initializer=tf.keras.initializers.get(config.get('bias_initializer', 'Zeros')),
                kernel_regularizer=tf.keras.regularizers.get(config.get('kernel_regularizer')),
                bias_regularizer=tf.keras.regularizers.get(config.get('bias_regularizer')),
                activity_regularizer=tf.keras.regularizers.get(config.get('activity_regularizer')),
                kernel_constraint=tf.keras.constraints.get(config.get('kernel_constraint')),
                bias_constraint=tf.keras.constraints.get(config.get('bias_constraint'))
            )
    
    @staticmethod
    def process_image(img):
        """image preprocessing for model prediction"""
        img = img.convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    @property
    def get_model(self):
        return self.model


if __name__ == "__main__":
    model_path = "path_to_model"
    img_path = "path_to_img"

    wrapper_obj = model_wrapper(model_path)
    artwork_model = wrapper_obj.get_model

    with Image.open(img_path) as img_obj:
        img = wrapper_obj.process_image(img_obj)

    prediction = artwork_model.predict(img)
    print(f"Class prediction: {prediction}")
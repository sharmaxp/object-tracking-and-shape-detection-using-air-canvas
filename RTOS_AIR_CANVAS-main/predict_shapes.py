import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
from PIL import Image

with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
# Load your model or perform other operations
# Load the trained model
    model = tf.keras.models.load_model('shape_classifier_model3_transfer_learning.h5') #model3
    # model = tf.keras.models.load_model('shape_classifier_model3_transfer_learning.h5') #model2
    # model = tf.keras.models.load_model('model1-shapeprediction.h5') #model1
    

def predict_shape(image_path):
    # Register the custom object using custom_object_scope
    global model
    

    threshold = {
        "circle": 0.92,
        "triangle": 0.97,
        "square": 0.5,
        "line": 0.92
    }

    # Make predictions on a new image
    # new_image_path = 'saved/saved.jpg'
    new_image = Image.open(image_path)
    new_image = new_image.resize((224, 224))
    new_image_array = np.array(new_image)
    new_image_array = np.expand_dims(new_image_array, axis=0)
    prediction = model.predict(new_image_array)
    predicted_class = np.argmax(prediction[0])
    class_confidence = prediction[0][predicted_class]

    class_probabilities = prediction[0]
    highest_probability = class_probabilities[predicted_class]

    shapes_list = ['circle','line','square','triangle']

    # class_name = shapes_list[predicted_class]

    # if class_name == None:
    #     print("The shape is not recognized.")
    # else:
    #     print(f"It's a {shapes_list[predicted_class]}!")

    # return class_name


    for shape_class, prob in zip(shapes_list, prediction[0]):
        print(f"Probability of {shape_class}: {prob:.4f}")

    if class_confidence >= threshold[shapes_list[predicted_class]]:
        class_name = shapes_list[predicted_class]
        print(f"The image is classified as: {class_name}")
    else:
        print("The image does not belong to any class.")
        class_name=None


    # return class_name
    return class_name, highest_probability

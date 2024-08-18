import argparse
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# image process function
def process_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image

# pridection function
def predict(image_path, model, top_k):
    #processs and load the image
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)

    # top K probs and class indices
    top_values, top_indices = tf.math.top_k(predictions, k=top_k)
    probs = top_values.numpy()[0]
    classes = top_indices.numpy()[0]

    classes = [str(class_index) for class_index in classes]

    return probs, classes

class MyModel(tf.keras.Model):
    def __init__(self, feature_extractor, classes_num):
        super(MyModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.dense = tf.keras.layers.Dense(classes_num, activation='softmax')

    def call(self, inputs):
        x = self.feature_extractor(inputs)
        return self.dense(x)

def load_model(model_path):
    # the model architecture rebuild
    IMAGE_SIZE = 224
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), trainable=False)
    model = MyModel(feature_extractor, classes_num=102)

    # weights load
    model.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))
    model.load_weights(model_path)

    return model


def main():
    #handle command-line arguments
    parser = argparse.ArgumentParser(description='Flower image classifier')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, help='Path to the saved model (HDF5 format)')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file map labels to flower names')

    args = parser.parse_args()

    model = load_model(args.model_path)

    # predictions
    probs, classes = predict(args.image_path, model, args.top_k)

    # load category if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        classes = [class_names.get(str(cls), "Unknown") for cls in classes]

    print("Predicted Probabilities:", probs)
    print("Predicted Classes:", classes)

if __name__ == '__main__':
    main()

import argparse
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from PIL import Image
import json

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224,224))
    image = tf.cast(image, tf.float32)
    image /=255.0

    return image.numpy()

def predict(image_path, model, top_k):

    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)


    ps = model.predict(image)
    
    ps = ps[0]
    top_prop_predictions = []
    index = []

    for i in range(top_k):
        m = max(ps)
        index.append(np.where(ps == m)[0][0])
        top_prop_predictions.append(m)
        ps = np.delete(ps, index[i], axis=0)
    
    return top_prop_predictions, index

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', type=str, default="./label_map.json")

    args = parser.parse_args()

    
    model = tf.keras.models.load_model("my_model.h5", custom_objects={"KerasLayer": hub.KerasLayer})

    probs, classes = predict(args.image_path, model, args.top_k)


    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)

        labels = [class_names[str(i)] for i in classes]
    else:
        labels = classes

    for prob, label in zip(probs, labels):
        print(f"{label}: {prob:.4f}")

if __name__ == "__main__":
    main()

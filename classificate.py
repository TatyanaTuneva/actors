import numpy as np
import face_recognition
from PIL import ImageOps
import json
import tensorflow as tf
from tensorflow import keras
import requests
from bs4 import BeautifulSoup


with open('config/actress_actors.json', 'r') as fp:
    class_names = json.load(fp)

normalization_layer = keras.layers.Rescaling(1./255)
model = keras.models.load_model('config/model')


def cropp(img):
    width, height = img.size
    if width == height == 200:
        return img

    face = np.array(img)
    coordinates = face_recognition.face_locations(face)

    if len(coordinates) == 1:
        top, right, bottom, left = coordinates[0]
        crop = (left - left // 2, top - top // 2, right + (width - right) // 2, bottom + (height - bottom) // 2)
        img_crop = img.crop(crop)
        img_crop = ImageOps.fit(img_crop, (200, 200), method=0, bleed=0.0, centering=(0.5, 0.5))
        return img_crop
    else:
        return len(coordinates)


def predict(img):
    img = cropp(img)

    if type(img) == int:
        return img

    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = normalization_layer(img_array)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    ind = np.argpartition(score, -3)[-3:]
    return [{'score': score[i], 'name': class_names[i]} for i in ind]


def get_image_actor(actor):
    s = requests.Session()
    name = actor.replace(' ', '%20')
    page = s.get(f'https://yandex.ru/images/search?text={name}')
    soup = BeautifulSoup(page.text, "html.parser")
    text = soup.findAll(attrs={'class': 'serp-item__thumb justifier__thumb'})[0]
    url = "http:" + text.get('src')
    img_data = requests.get(url).content
    return img_data
import os
from os.path import exists
from io import BytesIO
import numpy
import numpy as np
from typing import Any, List, Tuple, Union
import pandas as pd
import matplotlib.image as mpimg
import seaborn as sns
from skimage import io
from skimage import data
from skimage.util import compare_images
import tensorflow as tf
tf.get_logger().setLevel('WARN')           # Suppress TensorFlow logging (2)
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetV2L
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pickle

import pathlib
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import warnings
warnings.filterwarnings("ignore")

#object classification constants
PROJECT_DATA_PATH = './models/'
PROJECT_MODEL_PATH = PROJECT_DATA_PATH + 'prod/'
#image expected size
#IMAGE_HEIGHT = 224 #b0
#IMAGE_WIDTH = 224 #b0
IMAGE_HEIGHT = 480 #l
IMAGE_WIDTH = 480 #l

modelname = "EfficientNetV2L"

#object detection constants
OD_LABEL_FILENAME = 'mscoco_label_map.pbtxt'
OD_MODEL_DATE = '20200711'
OD_MODEL_NAME = 'centernet_hg104_1024x1024_coco17_tpu-32'

def InitModels():

    PATH_TO_MODEL_DIR = download_model(OD_MODEL_NAME, OD_MODEL_DATE)
    PATH_TO_LABELS = download_labels(OD_LABEL_FILENAME)
    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    epoch, model, opt = load_model_data(PROJECT_MODEL_PATH + modelname + "/model.hdf5", PROJECT_MODEL_PATH + modelname + "/model.pkl")
    labels = load_labels(PROJECT_MODEL_PATH + modelname)

    model.compile(
        optimizer=tf.keras.optimizers.Adam.from_config(opt),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return detect_fn, category_index, labels, model


#object detection plus classification driver method.
def find(image_np, detect_fn, category_index, labels, model):
    """Find objects in an image, and if the object found is a bird classify the bird's species

    Args:
      image_np - a numpy array of the image data.

    Returns:
      {
        'ImageWithBoxes': PIL image with bounding boxes, 
        'Predictions' : {
                            'Class': the object dection class name, 
                            'Score': Percentage Probablity score of object dection class, 
                            'Location': (bottom pixel, left most pixel, top pixel, right most pixel)
                            'Species' = Bird Speices identified
                            'Probablity' = Percentage Probablity score for the bird to be that species}
      }
    """

    imgHeight = image_np.shape[0]
    imgWidth = image_np.shape[1]

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np.copy(),
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.60,
      agnostic_mode=False)

    possible_idx = []
    for idx in range(0, len(detections['detection_scores'])):
        if detections['detection_scores'][idx] > .75:
            (ymin, xmin, ymax, xmax) = detections['detection_boxes'][idx]
            possible_idx.append({'Class': category_index[detections['detection_classes'][idx]]['name'], 'Score': round(detections['detection_scores'][idx]*100), 'Location': (round(ymin * imgHeight), round(xmin * imgWidth), round(ymax * imgHeight), round(xmax * imgWidth))})
    
    for possible in possible_idx:
        if possible['Class'] == 'bird':
            just_bird = image_np[possible['Location'][0]:possible['Location'][2], possible['Location'][1]:possible['Location'][3]]
            im = PIL.Image.fromarray(numpy.uint8(just_bird)).resize((IMAGE_HEIGHT, IMAGE_WIDTH))
            img_array = keras.preprocessing.image.img_to_array(im)
            img_array = tf.expand_dims(im, 0)  # Create batch axis
            label, score = predict(modelname, img_array, model, labels)
            possible['Species'] = label
            possible['Probablity'] = score
            
    return {'ImageWithBoxes': image_np_with_detections, 'Predictions' : possible_idx}

def predict(modelname, img_array, model, labels):
    """Classify the bird species.

    Args:
      modelname - modelname for the model to be loaded/used.
      img_array - Keras image data object array.

    Returns:
      (
          bird species label,
          percentage score for that species
      )
    """

    predictions = model.predict(img_array)
    label = 'unknown'
    idx = 0
    for x in predictions[0]:
        if x > 0.8:
            score = x
            break # I do this to just return the fastest possible ideally we would loop through them all and find the highest score and return that.
        idx = 1+idx
    label = labels[idx]

    return label, score*100

def load_model_data(model_path, opt_path):
    """Load saved model data for use.

    Args:
        mdoel_path - Path to the model directory to load the model.hdf5 from.
        opt_path - Path to the model directory to load the model.plk from.

    Returns:
        epoch - int for the epoch of the model loaded during the training.
        model - the model to be used.
        opt - the optimizer config to be used when compiling the model.

    """

    model = load_model(model_path)
    with open(opt_path, 'rb') as fp:
      d = pickle.load(fp)
      epoch = d['epoch']
      opt = d['opt']
      return epoch, model, opt

def load_labels(model_path):
    """Load the labels file to map the classified bird species to the id returned.

    Args:
        model_path - Path to the model directory to load the labels.txt from.
    
    Returns:
        labels - a list of labels in order to use the index for finding the name to display.
    
    """

    labels = []
    # open file and read the content in a list
    with open(model_path + '/labels.txt', 'r') as file:
        for line in file:
            # remove linebreak which is the last character of the string
            label = line[:-1]
            # add item to the list
            labels.append(label)
    return labels

## Object detection model download bits.
# Download and extract model
def download_model(model_name, model_date):
    """Boiler plate download object detection pretrained models.

    Args:
        model_name - name of the model file to download.
        model_date - the dated version to download.

    Returns:
        string - path to the model for loading and using in object detection.
    """
    if os.path.exists(os.path.expanduser('~') + f'/.keras/datasets/{model_name}/saved_model/saved_model.pb'):
        return os.path.expanduser('~') + f'/.keras/datasets/{model_name}'

    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'

    model_dir = tf.keras.utils.get_file(fname=model_name,
                                    origin=base_url + model_date + '/' + model_file,
                                    untar=True)
    return str(model_dir)

# Download labels file
def download_labels(filename):
    """Boiler plate download object detection pretrained labels to match the pretrained model.

    Args:
        filename - name of the labels file to download.

    Returns:
        string - path to the downlaoded lables.
    """

    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'

    if os.path.exists(os.path.expanduser('~') + '/.keras/datasets/' + filename):
        return os.path.expanduser('~') + '/.keras/datasets/' + filename

    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)

#Convert the pil image to numpy array for processing.
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """

    return np.array(Image.open(path))

def load_image_into_numpy_array_bytes(data):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """

    return np.array(Image.open(BytesIO(data)))
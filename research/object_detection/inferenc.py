import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import time
import numpy as np
from PIL import Image
from object_detection.utils import visualization_utils as viz_utils

# PROVIDE PATH TO INPUT IMAGE DIRECTORY
INPUT_IMAGE_DIR = 'C:/Users/Admin/Downloads/Hard_validation_sample_500_v3 - Copy'

# PROVIDE PATH TO OUTPUT IMAGE DIRECTORY
OUTPUT_IMAGE_DIR = 'S:/Hard_validation_sample_500_v3 - Copy'

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = 'S:/phones/models/research/object_detection/efficientdet_d0_coco17_tpu-32'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL
print('Loading model...', end='')
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR + "/saved_model")
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# MODIFY LABEL MAP TO CHANGE THE LABEL
category_index = {1: {'id': 1, 'name': 'phone'}}

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into TensorFlow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = Image.open(path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')  # Convert RGBA to RGB
    return np.array(image)

def process_image(image_path):
    print('Running inference for {}... '.format(image_path), end='')
    # Load the image
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=MIN_CONF_THRESH,
          agnostic_mode=False)

    output_image_path = os.path.join(OUTPUT_IMAGE_DIR, os.path.relpath(image_path, INPUT_IMAGE_DIR))
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, cv2.cvtColor(image_with_detections, cv2.COLOR_RGB2BGR))
    print('Saved output image to:', output_image_path)

# Process each image in the input directory and its subdirectories
for root, dirs, files in os.walk(INPUT_IMAGE_DIR):
    for filename in files:
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            image_path = os.path.join(root, filename)
            process_image(image_path)

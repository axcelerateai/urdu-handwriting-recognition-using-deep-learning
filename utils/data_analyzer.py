import numpy as np
import nltk
import os
import cv2
import collections

from utils.data_utils import UrduTextReader

def _list_all_files(path):
    files = [os.path.join(path, f) for f in os.listdir(path)]

    return files

def _get_image_shape(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    return image.shape

def get_images_mean(path):
    images = _list_all_files(path)
    height = []
    width = []
    for i, image in enumerate(images):
        s = _get_image_shape(image)
        height.append(s[0])
        width.append(s[1])

        if i % 500 == 0:
            print("Processed ", i, " images")

    mean_height = sum(height)/len(height)
    mean_width = sum(width)/len(width)
    max_height = max(height)
    max_width = max(width)
    min_height = min(height)
    min_width = min(width)

    print("Total Number of Images: ", len(images))
    print("Mean height = ", mean_height, "| Mean width = ", mean_width, "| Max height = ", max_height, "| Max width = ", max_width,
          "| Min height = ", min_height, "| Min width = ", min_width)

def get_grams_info(path):
    reader = UrduTextReader()
    text = []
    with open(path, "r", encoding='utf-8') as f:
        for row in f:
            text.append(reader.clean(row))

    grams_1 = []
    grams_2 = []
    grams_3 = []
    for row in text:
        grams_1.extend(list(nltk.ngrams(row, 1)))
        grams_2.extend(list(nltk.ngrams(row, 2)))
        grams_3.extend(list(nltk.ngrams(row, 3)))

    c_grams_1 = collections.Counter(grams_1)
    c_grams_2 = collections.Counter(grams_2)
    c_grams_3 = collections.Counter(grams_3)

    print("Unigrams: ", len(c_grams_1))
    print("Bigrams:  ", len(c_grams_2))
    print("Trigrams: ", len(c_grams_3))

"""
This is the only file that depends on the structure of the data folder. It assumes the following structure:
    -> data_folder
        -> labels
            -> gt_char.csv      (contains labels for character mode)
            -> gt_lig.csv       (contains labels for ligature mode)
            -> constants.csv    (contains vocab sizes for both character and ligature modes)
            -> lt_char.csv      (mappings from numbers to characters for character mode)
            -> lt_lig.csv       (mappings from numbers to ligatures for ligature mode)
        -> images               (contains images in either jpg, png or jpeg formats)

Each line in gt_char.csv and gt_lig.csv is of the following format:
    name.*,[0,10,1,4]
and has a corresponding image named name.* in the images folder. The labels can be of any length.

Other structures may be supported by modifying load_data and get_lookup_table functions.
"""

from __future__ import print_function

from utils.image_utils import *

import os
import numpy as np
import csv

def _load_images(filenames, image_size, flip_image=True, buckets=1):
    images, seq_len, bucket_indices = handle_training(filenames, image_size, flip_image=flip_image, buckets=buckets)

    return images, seq_len, bucket_indices

def _size_alright(directory, img_width=None, image_range=None):
    if image_range == None:
        return True

    image = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
    if image.shape[1] <= img_width + img_range and image.shape[1] >= img_width - img_range:
        return True

    return False

def _get_image_path(path):
    if os.path.isfile(path + ".png"):
        return path + ".png"
    elif os.path.isfile(path + ".jpg"):
        return path + ".jpg"
    elif os.path.isfile(path + ".jpeg"):
        return path + ".jpeg"
    else:
        return None

def load_gt(gt_directory, image_directory=None, image_width=None, image_range=None, size=None):
    gt = []
    filenames = []
    count = 0

    with open(gt_directory, 'r', encoding='utf8') as gt_file:
        text = csv.reader(gt_file, quoting=csv.QUOTE_NONE)
        for row in text:
            if image_directory is None:
                is_row_valid = True
            else:
                image_name = row[0].split('.')[0]
                path = _get_image_path(os.path.join(image_directory, image_name))
                is_row_valid = path is not None and _size_alright(path, image_width, image_range)

            if is_row_valid:
                sub_row = row[1:]
                sub_row[0]  = sub_row[0].replace("[",'')
                sub_row[0]  = sub_row[0].replace('"','')
                sub_row[-1] = sub_row[-1].replace("]",'')
                sub_row[-1] = sub_row[-1].replace('"','')
                gt.append([int(num) for num in sub_row])
                
                if image_directory is not None:
                    filenames.append(path)
                
                count += 1

            if size != None:
                if count >= size:
                    break

    return gt, filenames

def _bucket_gt(gt, num_buckets, bucket_indices):
    y_seq_len = [len(g) for g in gt]
 
    if bucket_indices is None:
        y = [gt]
        y_seq_len = [np.array(y_seq_len)]

    else:
        y = [[] for _ in range(num_buckets)]
        seq_len = [[] for _ in range(num_buckets)]

        for i, j in enumerate(bucket_indices):
            y[j-1].append(np.array(gt[i]))
            seq_len[j-1].append(np.array(y_seq_len[i]))

        for i in range(num_buckets):
            y[i] = np.array(y[i])
            seq_len[i] = np.array(seq_len[i])

        y_seq_len = seq_len

    return np.array(y), np.array(y_seq_len)

# load the entire data
def load_data(char_or_lig, data_folder, image_size, size=None, image_width_range=None, flip_image=True, buckets=1):
    image_directory = os.path.join(data_folder, "images", "")
    gt_directory = os.path.join(data_folder, "labels", 'gt_' + char_or_lig + '.csv')

    groundtruths, filenames = load_gt(gt_directory,
                                      image_directory=image_directory,
                                      image_width=image_size[0],
                                      image_range=image_width_range,
                                      size=size)

    images, images_seq_len, bucket_indices = _load_images(filenames,
                                                          image_size,
                                                          flip_image=flip_image,
                                                          buckets=buckets)

    y, y_seq_len = _bucket_gt(groundtruths, buckets, bucket_indices)

    return images, images_seq_len, y, y_seq_len

def prepare_dataset(dataset, r=1.0, shuffle=True, all_indices=None):
    # Prepare 'dataset' for the minibatches function.
    # Each element in 'dataset' is a list of size 'num_buckets'.
    
    # If r != 1, then divide each bucket into two sets. Used to prepare test and validation
    # sets. Dividing each bucket separately makes the validation set more balanced.
 
    restoring_previous = False if all_indices is None else True

    num_buckets = dataset[0].shape[0]
    train_dataset = [[] for _ in range(num_buckets)]
    val_dataset = [[] for _ in range(num_buckets)]

    to_remove = []
    if not restoring_previous:
        all_indices = []
    for i in range(num_buckets):
        num_train = dataset[0][i].shape[0]

        if restoring_previous:
            indices = all_indices[i]
        else:
            indices = np.arange(num_train)
            if shuffle:
                np.random.shuffle(indices)
            all_indices.append(indices)

        split_at = int(np.ceil(r*num_train))
        indices_train = indices[:split_at]
        indices_val = indices[split_at:]

        train_dataset[i] = [d[i][indices_train] for d in dataset]
        if split_at == num_train:
            to_remove.append(i)
        else:
            val_dataset[i] = [d[i][indices_val] for d in dataset]

    val_dataset = [vd for i, vd in enumerate(val_dataset) if not i in to_remove] if r != 1 else []

    return train_dataset, val_dataset, all_indices

def minibatches(dataset, batch_size, shuffle=True):
    for bucket in dataset:
        num_examples = bucket[0].shape[0]
        indices = np.arange(0, num_examples)
 
        if shuffle:
            np.random.shuffle(indices)
 
        for minibatch_start in np.arange(0, len(indices), batch_size):
            minibatch_indices = indices[minibatch_start:minibatch_start+batch_size] 
            yield [b[minibatch_indices] for b in bucket]

def _get_buckets(seq_len, num_buckets=2):
    min_seq_len = min(seq_len)
    max_seq_len = max(seq_len)
    bins = np.arange(min_seq_len, max_seq_len, (max_seq_len-min_seq_len)/num_buckets)
    bucket_indices = np.digitize(seq_len, bins)
    for i in range(num_buckets):
        yield np.array([j for j, x in enumerate(bucket_indices) if x==(i+1)])

class UrduCharacters():
    def sent_enders(self):
        # ?-!{\n}
        return [1567, 1748, 33]

    def line_breakers(self):
        # {\n}{Data Line Escape (DLE)}
        return [10, 65279]

    def spacers(self):
        #  {Data Line Escape (DLE)}
        return [32, 65279]

    def garbage(self):
        # {Zero-Order Byte Mark (BOM)}{Zero Width Non-joiner}{Tab}{Arabic Footer}
        return [65279, 8204, 9, 1538]

    def non_joiners(self):
        #  ادڈذرڑزژےوں
        return [32, 1575, 1583, 1672, 1584, 1585, 1681, 1586, 1688, 1746, 1608, 1722]

    def isolators(self):
        # ءۓؤآۃأ
        # ﷻﷺ،۔؟؛'!/|()\'\"“”.:*-_
        # ۱۲۳۴۵۶۷۸۹٩٧٤٨۲٦۰٥٧٢٠١٣
        # 0123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz
        # {foot_note}
        return [1569, 1747, 1572, 1570, 1731, 1571, #
                65019, 65018, 1548, 1748, 1567, 1563, 39, 33, 47, 124, 40, 41, 39, 34, 8220, 8221,
                46, 58, 42, 45, 95, #
                1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1641, 1639, 1636, 1640, 1778,
                1638, 1776, 1637, 1639, 1634, 1632, 1633, 1635, #
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 97, 66, 98, 67, 99, 68, 100, 69, 101, 70,
                102, 71, 103, 72, 104, 73, 105, 74, 106, 75, 107, 76, 108, 77, 109, 78, 110, 79, 111,
                80, 112, 81, 113, 82, 114, 83, 115, 84, 116, 85, 117, 86, 118, 87, 119, 88, 120, 89,
                121, 90, 122, #
                1528] #

    def enders(self):
        return self.non_joiners() + self.isolators()

    def sent_extenders(self):
        # May come after sent_enders but are still part of the same sentence
        # “?-!
        return [8220, 1567, 1748, 33]

    def lig_extenders(self):
        # ''''''''
        return [1619, 1611, 1617, 1616, 1648, 1552, 1555, 1614]

class UrduTextReader():
    def __init__(self, filename=None):
        self.characters = UrduCharacters()
        if filename is not None:
            with open(filename, "r", encoding='utf-8') as f:
                self.text = self.clean(f.read())

    def clean(self, text):
        len_text = len(text)
        cleaned_text = ""
        i = 0
        while True:
            if not ord(text[i]) in self.characters.garbage():
                # {alif madda->alif_madda}
                if i != len_text-1 and ord(text[i]) == 1575 and ord(text[i+1]) == 1619:
                    cleaned_text += chr(1570)
                    i += 1
                elif i != len_text-1 and ord(text[i]) == 34:
                    cleaned_text += chr(39) + chr(39)
                elif ord(text[i]) in self.characters.line_breakers():
                    if i != len_text-1 and not ord(text[i+1]) in self.characters.sent_enders():
                        cleaned_text += chr(32) # add space
                else:
                    cleaned_text += text[i]
            i += 1
            if i == len_text:
                break
        return cleaned_text

    def sents(self):
        sents = self.split_str_at(self.text,
                                  self.characters.sent_enders(),
                                  extenders=self.characters.sent_extenders(),
                                  spacers=None,
                                  isolated=None,
                                  isolated_extenders=None)
        sents = [self.ligs(sent) for sent in sents]

        return sents

    def ligs(self, sent):
        split = self.split_str_at(sent,
                                  self.characters.non_joiners(),
                                  extenders=self.characters.lig_extenders(),
                                  spacers=self.characters.spacers(),
                                  isolated=self.characters.isolators(),
                                  isolated_extenders=None)

        return split

    def split_str_at(self, string, split_at, extenders=None, spacers=None, isolated=None, isolated_extenders=None):
        return [s for s in self._split_str_at(string,
                                              split_at,
                                              extenders=extenders,
                                              spacers=spacers,
                                              isolated=isolated,
                                              isolated_extenders=isolated_extenders)
                if len(s) != 0]

    def _split_str_at(self, string, split_at, extenders=None, spacers=None, isolated=None, isolated_extenders=None):
        last_cut = -1
        i = 0
        while True:
            if isolated is not None and ord(string[i]) in isolated:
                if isolated_extenders is not None:
                    while i != len(string)-1 and ord(string[i+1]) in isolated_extenders:
                        i += 1
                yield string[last_cut+1:i]
                yield str(string[i])
                last_cut = i
            elif spacers is not None and ord(string[i]) in spacers:
                yield string[last_cut+1:i]
                last_cut = i
            elif ord(string[i]) in split_at:
                if extenders is not None:
                    while i != len(string)-1 and ord(string[i+1]) in extenders:
                        i += 1
                yield string[last_cut+1:i+1]
                last_cut = i

            i += 1
            if i == len(string):
                break

        if i != last_cut:
            yield string[last_cut+1:]

    def strip(self, string, remove):
        return_string = ""
        for s in string:
            if not ord(s) in remove:
                return_string += s

        return return_string

# This is specifically for converting numbers back to Urdu text. May not work for other
# languages.
def convert_to_urdu(output, data_folder):
    lt = get_lookup_table(data_folder)
    urdu_output = [lt[output[i]] for i in range(len(output)-1, -1, -1)]

    join_char = ""
    for i in range(len(urdu_output)-1, -1, -1):
        join_char += urdu_output[i][0]
        if urdu_output[i][2:] == 'final' or urdu_output[i][2:] == 'isolated':
            join_char += ' '

    return urdu_output, join_char

def get_lookup_table(data_folder):
    lt = {}
    lt_file = os.path.join(data_folder, "labels/lt_char.csv")
    with open(lt_file, 'r', encoding='utf8') as f:
        text = csv.reader(f, quotechar=None)
        for row in text:
            if len(row) == 2:
                lt[int(row[1])] = row[0]
            else:
                # We must have a comma in this field in this case
                lt[int(row[2])] = ',' + row[1]
    return lt 

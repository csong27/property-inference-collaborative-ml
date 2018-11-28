from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from scipy.misc import imread, imresize

import pandas as pd
import numpy as np
import warnings

import os


LFW_DIR = '/home/song/scikit_learn_data/lfw_home/lfw_funneled/'


def load_lfw(attr_type='gender', transpose=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = fetch_lfw_people(color=True)

    images = data.images  # * 255
    names = [data.target_names[i] for i in data.target]
    name_attr = load_attribute(attr_type)
    labels = []
    indices = []
    for i in xrange(len(names)):
        name = names[i]
        if name in name_attr:
            labels.append(name_attr[name])
            indices.append(i)

    X = images[indices].astype(np.float32)
    if transpose:
        X = X.transpose(0, 3, 1, 2)
    y = np.asarray(labels, dtype=np.int32)
    train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    return train_x, train_y, test_x, test_y


def load_lfw_raw(transpose=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = fetch_lfw_people(color=True)

    images = data.images
    names = [data.target_names[i] for i in data.target]

    X = images.astype(np.int32)
    if transpose:
        X = X.transpose(0, 3, 1, 2)

    return X, names


def load_attribute(attr_type='gender'):
    attr = pd.read_csv('lfw_attributes.txt', delimiter='\t')
    names = np.asarray(attr['person'])
    if attr_type == 'gender':
        gender = np.asarray(attr['Male'])
        gender = np.sign(gender)
        gender[gender == -1] = 0
        name_to_gender = dict(zip(names, gender))
        return name_to_gender
    elif attr_type == 'race':
        race = np.asarray(attr[['Asian', 'White', 'Black']])
        name_to_race = dict()
        for i, name in enumerate(names):
            if name not in name_to_race:
                name_to_race[name] = (np.max(race[i]), np.argmax(race[i]))
            else:
                max_score = np.max(race[i])
                if max_score > name_to_race[name][0]:
                    if name_to_race[name][1] != np.argmax(race[i]):
                        continue
                    name_to_race[name] = (np.max(race[i]), np.argmax(race[i]))

        for name in name_to_race:
            name_to_race[name] = name_to_race[name][1]

        return name_to_race
    elif attr_type == 'glasses':
        race = np.asarray(attr[['Eyeglasses', 'Sunglasses', 'No Eyewear']])
        name_to_race = dict()
        for i, name in enumerate(names):
            if name not in name_to_race:
                name_to_race[name] = (np.max(race[i]), np.argmax(race[i]))
            else:
                max_score = np.max(race[i])
                if max_score > name_to_race[name][0]:
                    if name_to_race[name][1] != np.argmax(race[i]):
                        continue
                    name_to_race[name] = (np.max(race[i]), np.argmax(race[i]))

        for name in name_to_race:
            name_to_race[name] = name_to_race[name][1]

        return name_to_race
    elif attr_type == 'age':
        age = np.asarray(attr[['Baby', 'Child', 'Youth', 'Middle Aged', 'Senior']])
        age = np.argmax(age, 1)
        name_to_age = dict(zip(names, age))
        return name_to_age
    elif attr_type == 'smile':
        smile = np.asarray(attr['Smiling'])
        smile = np.sign(smile)
        smile[smile == -1] = 0
        name_to_smile = dict(zip(names, smile))
        return name_to_smile
    else:
        raise ValueError(attr_type)


def save_lfw(slice_=(slice(70, 195), slice(78, 172)), resize=0.5):
    attr = pd.read_csv('./data/lfw_attributes.txt', delimiter='\t')
    names = np.asarray(attr['person'])
    img_num = np.asarray(attr['imagenum'])

    default_slice = (slice(0, 250), slice(0, 250))
    slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))
    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)

    imgs = np.zeros((len(names), h, w, 3), dtype=np.uint8)
    i = 0
    for name, num in zip(names, img_num):
        name = name.replace(' ', '_')
        img_path = os.path.join(LFW_DIR, name, '{}_{}.jpg'.format(name, str(num).zfill(4)))
        img = imread(img_path)[slice_]
        img = imresize(img, resize)
        imgs[i] = img
        i += 1

    np.savez('./data/lfw_images.npz', imgs)


def load_lfw_binary_attr(attr_type='gender'):
    attr = pd.read_csv('./data/lfw_attributes.txt', delimiter='\t')
    if attr_type == 'gender':
        gender = np.asarray(attr['Male'])
        gender = np.sign(gender)
        gender[gender == -1] = 0
        # print len(gender), np.mean(gender == 1), np.mean(gender == 0)
        return dict(zip(range(len(gender)), gender))
    elif attr_type == 'smile':
        smile = np.asarray(attr['Smiling'])
        smile = np.sign(smile)
        smile[smile == -1] = 0
        return dict(zip(range(len(smile)), smile))
    else:
        raise ValueError(attr_type)


def load_lfw_multi_attr(attr_type='race', thresh=-0.1):
    attr = pd.read_csv('./data/lfw_attributes.txt', delimiter='\t')

    if attr_type == 'race':
        attr = np.asarray(attr[MULTI_ATTRS['race']])
    elif attr_type == 'glasses':
        attr = np.asarray(attr[MULTI_ATTRS['glasses']])
    elif attr_type == 'age':
        attr = np.asarray(attr[MULTI_ATTRS['age']])
    elif attr_type == 'hair':
        attr = np.asarray(attr[MULTI_ATTRS['hair']])
    else:
        raise ValueError(attr_type)

    indices = []
    labels = []
    for i, a in enumerate(attr):
        if np.max(a) < thresh:  # score too low for an attribute
            continue
        indices.append(i)
        labels.append(np.argmax(a))

    return dict(zip(indices, labels))


BINARY_ATTRS = {'gender': ['Female', 'Male'],
                'smile': ['Not Smiling', 'Smiling']}

MULTI_ATTRS = {'race': ['Asian', 'White', 'Black'],
               'glasses': ['Eyeglasses', 'Sunglasses', 'No Eyewear'],
               'age': ['Baby', 'Child', 'Youth', 'Middle Aged', 'Senior'],
               'hair': ['Black Hair', 'Blond Hair', 'Brown Hair', 'Bald']}


def load_lfw_attr(attr='gender'):
    return load_lfw_binary_attr(attr) if attr in BINARY_ATTRS else load_lfw_multi_attr(attr)


def load_lfw_with_attrs(attr1='gender', attr2=None):
    with np.load('./data/lfw_images.npz') as f:
        imgs = f['arr_0'].transpose(0, 3, 1, 2)

    index_label_1 = load_lfw_attr(attr1)
    if attr2 is None:
        indices = np.sort(index_label_1.keys())
        imgs = imgs[indices] / np.float32(255.0)
        labels = np.asarray([index_label_1[i] for i in indices], dtype=np.int32)
        return imgs, labels

    index_label_2 = load_lfw_attr(attr2)
    common_indices = np.intersect1d(index_label_1.keys(), index_label_2.keys())
    imgs = imgs[common_indices] / np.float32(255.0)
    labels1 = np.asarray([index_label_1[i] for i in common_indices], dtype=np.int32)
    labels2 = np.asarray([index_label_2[i] for i in common_indices], dtype=np.int32)

    return imgs, labels1, labels2

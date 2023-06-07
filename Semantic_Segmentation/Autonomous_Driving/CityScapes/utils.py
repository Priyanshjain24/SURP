import os
import cv2
import numpy as np
import keras.backend as K
import segmentation_models as sm
from glob import glob
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical

Label = namedtuple('Label', ['name', 'color', 'value'])

label_defs = [
    Label('unlabeled',     (0,     0,   0),  0),
    Label('dynamic',       (111,  74,   0),  1),
    Label('ground',        ( 81,   0,  81),  2),
    Label('road',          (128,  64, 128),  3),
    Label('sidewalk',      (244,  35, 232),  4),
    Label('parking',       (250, 170, 160),  5),
    Label('rail track',    (230, 150, 140),  6),
    Label('building',      ( 70,  70,  70),  7),
    Label('wall',          (102, 102, 156),  8),
    Label('fence',         (190, 153, 153),  9),
    Label('guard rail',    (180, 165, 180), 10),
    Label('bridge',        (150, 100, 100), 11),
    Label('tunnel',        (150, 120,  90), 12),
    Label('pole',          (153, 153, 153), 13),
    Label('traffic light', (250, 170,  30), 14),
    Label('traffic sign',  (220, 220,   0), 15),
    Label('vegetation',    (107, 142,  35), 16),
    Label('terrain',       (152, 251, 152), 17),
    Label('sky',           ( 70, 130, 180), 18),
    Label('person',        (220,  20,  60), 19),
    Label('rider',         (255,   0,   0), 20),
    Label('car',           (  0,   0, 142), 21),
    Label('truck',         (  0,   0,  70), 22),
    Label('bus',           (  0,  60, 100), 23),
    Label('caravan',       (  0,   0,  90), 24),
    Label('trailer',       (  0,   0, 110), 25),
    Label('train',         (  0,  80, 100), 26),
    Label('motorcycle',    (  0,   0, 230), 27),
    Label('bicycle',       (119,  11,  32), 28)]


def load_data(images_root, labels_root, image_shape, sample_name):
    images = []
    masks = []
    image_sample_root = images_root + '/' + sample_name
    image_root_len = len(image_sample_root)
    label_sample_root = labels_root + '/' + sample_name
    image_files = glob(image_sample_root + '/**/*png')
    for f in image_files:
        f_relative = f[image_root_len:]
        f_dir = os.path.dirname(f_relative)
        f_base = os.path.basename(f_relative)
        f_base_gt = f_base.replace('leftImg8bit', 'gtFine_color')
        f_label = label_sample_root + f_dir + '/' + f_base_gt
        if os.path.exists(f_label):
            images.append(cv2.cvtColor(cv2.resize(cv2.imread(f), image_shape), cv2.COLOR_BGR2RGB))
            masks.append(cv2.cvtColor(cv2.resize(cv2.imread(f_label), image_shape), cv2.COLOR_BGR2RGB))
    return images, masks

def scale_image_pixels(images):
    scaler = MinMaxScaler()
    new = []
    for i,single_patch_img in enumerate(images):
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        new.append(single_patch_img)
    return new

def rgb2label(dataset, num_classes):
    temp = []
    
    for label in dataset:   
        label_seg = np.zeros(label.shape, dtype=np.uint8)

        for i in range(num_classes):
            label_seg [np.all(label==label_defs[i][1],axis=-1)] = label_defs[i][2]

        label_seg = label_seg[:,:,0] 
        temp.append(label_seg)
        
    return np.array(temp)

def to_categories(dataset, num_classes):
    labels = np.expand_dims(dataset, axis=3)
    return to_categorical(labels, num_classes=num_classes)

def load_train_val_data(data_dir, target_shape):
    
    images_root = data_dir + '/leftImg8bit'
    labels_root = data_dir + '/gtFine'
    num_classes = len(label_defs)

    train_images, train_masks = load_data(images_root, labels_root, target_shape, 'train')
    val_images, val_masks = load_data(images_root, labels_root, target_shape, 'val')

    train_images = scale_image_pixels(train_images)
    train_images = np.array(train_images)

    val_images = scale_image_pixels(val_images)
    val_images = np.array(val_images)

    train_masks = np.array(train_masks)
    train_masks = rgb2label(train_masks, num_classes)
    train_masks = to_categories(train_masks, num_classes)

    val_masks = np.array(val_masks)
    val_masks = rgb2label(val_masks, num_classes)
    val_masks = to_categories(val_masks, num_classes)

    return num_classes, train_images, train_masks, val_images, val_masks

def load_test_data(data_dir, target_shape):

    images_root = data_dir + '/leftImg8bit'
    labels_root = data_dir + '/gtFine'
    num_classes = len(label_defs)

    test_images, test_masks = load_data(images_root, labels_root, target_shape, 'test')

    test_images = scale_image_pixels(test_images)
    test_images = np.array(test_images)

    test_masks = np.array(test_masks)
    test_masks = rgb2label(test_masks, num_classes)
    test_masks = to_categories(test_masks, num_classes)

    return num_classes, test_images, test_masks

def label2rgb(dataset, num_classes, shape):
    temp = []
    
    for label in dataset:   
        label_seg = np.zeros(shape, dtype=np.uint8)

        for i in range(num_classes):
            label_seg [np.all(label==label_defs[i][2],axis=-1)] = label_defs[i][1]

        temp.append(label_seg)
        
    return np.array(temp)

def save_predictions(dataset, pred_path):

    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
        pass

    for i,pred in enumerate(dataset):
        cv2.imwrite(pred_path + '/' + "%03d" % i + '.jpg', pred)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def weights_list(dataset):
    temp = np.argmax(dataset, axis = -1)
    labels, counts = np.unique(temp, return_counts=True)
    weights_train = []

    for i in counts:
        weights_train.append( i / temp.size )

    return weights_train

def params(dataset):

    weights = weights_list(dataset)
    dice_loss = sm.losses.DiceLoss(class_weights=weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss) 
    metrics=['accuracy', jacard_coef]

    return total_loss, metrics



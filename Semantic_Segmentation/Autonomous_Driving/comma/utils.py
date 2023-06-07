import os
import cv2
import numpy as np
import keras.backend as K
import segmentation_models as sm
from glob import glob
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical, Sequence
from keras.metrics import MeanIoU

Label = namedtuple('Label', ['name', 'color'])

label_defs = [
    Label('road',           (  64,  32,  32 ) ), # road (all parts, anywhere nobody would look at you funny for driving)
    Label('lanes',          ( 255,   0,   0 ) ), # lane markings (don't include non lane markings like turn arrows and crosswalks)
    Label('undrivable',     ( 128, 128,  96 ) ), # undrivable
    Label('movable',        (   0, 255, 102 ) ), # movable (vehicles and people/animals)
    Label('my_car',         ( 204,   0, 255 ) )] # my car (and anything inside it, including wires, mounts, etc. No reflections)

def num_classes():
    return len(label_defs)

def scale_image_pixels(images):
    scaler = MinMaxScaler()
    new = []
    for i,single_patch_img in enumerate(images):
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        new.append(single_patch_img)
    return np.array(new)

def rgb2cat(dataset):
    temp = []
    dataset = np.array(dataset)
    
    for label in dataset:   
        label_seg = np.zeros(label.shape, dtype=np.uint8)

        for i, region in enumerate(label_defs):
            label_seg [np.all(label==region[1] , axis=-1)] = i

        label_seg = label_seg[:,:,0] 
        temp.append(label_seg)
        
    temp = np.expand_dims(temp, axis=3)
    return to_categorical(temp, num_classes=num_classes())

def get_data(dir, list, shape, indices, pic):
    pics = []
    for i in indices:
        pics.append(cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(dir, pic, list[i])), shape), cv2.COLOR_BGR2RGB))

    if (pic == 'imgs'):
        pics = scale_image_pixels(pics)

    elif (pic == 'masks'):
        pics = rgb2cat(pics)

    return pics

def cat2rgb(dataset, shape):
    temp = []
    
    dataset = np.expand_dims(np.argmax(dataset, axis = -1), axis=-1)

    for label in dataset:   
        label_seg = np.zeros(shape, dtype=np.uint8)

        for i, region in enumerate(label_defs):
            label_seg [np.all(label==i,axis=-1)] = region[1]

        temp.append(label_seg)
        
    return np.array(temp)

class DataGenerator(Sequence):

    def __init__(self, data_dir, shape, split, shuffle=False, batch_size=10):
        
        self.shape = shape
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dir = os.path.join(data_dir, split) if split != 'predict' else os.path.join(data_dir, 'retrain')
        self.split = split
        with open(os.path.join(self.dir, 'files'), 'r') as file:    
            self.list = [line.strip() for line in file.readlines()]
        self.indices = np.arange(len(self.list))

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(len(self.list) / self.batch_size)
    
    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        images = get_data(self.dir, self.list, self.shape, indices, pic='imgs')

        if ( (self.split == 'train') or (self.split == 'val') or (self.split == 'retrain') ):
            masks = get_data(self.dir, self.list, self.shape, indices, pic='masks')
            return images, masks
        
        elif (self.split == 'test' or self.split == 'predict'):
            return images

def save_predictions(dataset, data_dir, names):
    pred_path = os.path.join(data_dir, 'retrain/prediction/')

    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
        pass

    for i,pred in enumerate(dataset):
        cv2.imwrite(pred_path + names[i], cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def weights_list(weights_path):
    
    with open(weights_path, 'r') as file:
        weights = [float(line.strip().split()[2]) for line in file]
    
    return weights

def params(weights_path):

    weights = weights_list(weights_path)
    dice_loss = sm.losses.DiceLoss(class_weights=weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss) 
    metrics=['accuracy', jacard_coef]

    return total_loss, metrics

def load_test_masks(data_dir, shape):
    with open(os.path.join(data_dir, 'test/files',), 'r') as file:
        lines = [os.path.join(data_dir, 'test/masks',line.strip()) for line in file.readlines()]

    test_masks = [cv2.cvtColor(cv2.resize(cv2.imread(path), shape), cv2.COLOR_BGR2RGB) for path in lines]
    test_masks = rgb2cat(test_masks)
    test_masks = np.expand_dims(np.argmax(test_masks, axis=-1), axis = -1)

    return test_masks

def evaluate(ground_truth, prediction):
    IOU_keras = MeanIoU(num_classes=num_classes())  
    IOU_keras.update_state(ground_truth, prediction)
    print("Mean IoU = ", IOU_keras.result().numpy())
    

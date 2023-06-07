import os
from keras.callbacks import ModelCheckpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import squeezenet as sn
from unet import *
from utils import *


def train(n_labels, batch_size, epochs, saved_weight, input_shape, pre_weight, train_images, train_masks, validation_images, validation_masks, model_name):

    if not os.path.exists(saved_weight):
        os.makedirs(saved_weight)
        pass

    if (model_name == 'Squeeze'):
        context_Net = sn.squeeze_segNet(n_labels=n_labels, image_shape=input_shape, weights_path = pre_weight)
        model = context_Net.init_model()

    elif (model_name == 'U'):
        model = multi_unet_model(n_labels, IMG_HEIGHT=input_shape[1], IMG_WIDTH=input_shape[0], IMG_CHANNELS=input_shape[2])

    else:
        print("Specified model should be U-Net or Squeeze-Net")

    mid_model = saved_weight + '/CityScapes_' + model_name + '_mid.h5'
    fin_model = saved_weight + '/CityScapes_' + model_name + '_last.hdf5'
    total_loss, metrics = params(train_masks)

    chk = ModelCheckpoint(mid_model, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq="epoch")
    model.compile(optimizer='adam', loss = total_loss, metrics = metrics)
    model.fit(train_images, train_masks, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[chk], validation_data = (validation_images, validation_masks), shuffle = False)
    model.save(fin_model)

def main():
 
    data_dir = 'data'
    target_shape = (256, 128)
    input_shape = (256, 128, 3)
    batch_size = 10
    epochs = 40
    pre_weight = None #'data/sq_weight.h5'
    saved_weight = 'models'
    model_name = 'U' # U or Squeeze
    
    print("Loading Data")
    num_classes, train_images, train_masks, val_images, val_masks = load_train_val_data(data_dir, target_shape)
    print("Dataset Loaded")
    print("Starting Training")
    train(num_classes, batch_size, epochs, saved_weight, input_shape, pre_weight, train_images, train_masks, val_images, val_masks, model_name)
    print("Train Completed")

if __name__ == "__main__":
    main()







    

import os
from keras.callbacks import ModelCheckpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import squeezenet as sn
from unet import multi_unet_model
from utils import num_classes, params, DataGenerator

data_dir = 'data'
target_shape = (480, 288)
input_shape = (480, 288, 3)
batch_size = 10
epochs = 30
shuffle = False
pre_weight = None #'data/sq_weight.h5'
saved_weight = 'models'
model_name = 'U' # U or Squeeze
weights_path = 'weights'

n_labels = num_classes()
train = DataGenerator(data_dir=data_dir, shape=target_shape, split='train', batch_size=batch_size, shuffle=shuffle)
val = DataGenerator(data_dir=data_dir, shape=target_shape, split='val', batch_size=batch_size, shuffle=shuffle)

if not os.path.exists(saved_weight):
    os.makedirs(saved_weight)
    pass

if (model_name == 'Squeeze'):
    context_Net = sn.squeeze_segNet(n_labels=n_labels, image_shape=input_shape, weights_path = pre_weight)
    model = context_Net.init_model()

elif (model_name == 'U'):
    model = multi_unet_model(n_labels, IMG_HEIGHT=input_shape[1], IMG_WIDTH=input_shape[0], IMG_CHANNELS=input_shape[2])

else:
    print("Wrong Model Name")

mid_model = saved_weight + '/comma_' + model_name + '_mid.h5'
fin_model = saved_weight + '/comma_' + model_name + '_last.hdf5'
total_loss, metrics = params(weights_path)
chk = ModelCheckpoint(mid_model, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq="epoch")
model.compile(optimizer='adam', loss = total_loss, metrics = metrics)
model.summary()
model.fit(train, epochs=epochs, verbose=1, callbacks=[chk], validation_data = val)
model.save(fin_model)







    

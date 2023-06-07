import os
from keras.models import load_model
from utils import num_classes, params, DataGenerator, jacard_coef

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_dir = 'All_Data/June-5'
target_shape = (480, 288)
# input_shape = (480, 288, 3)
batch_size = 10
epochs = 1
shuffle = False
weights_path = 'weights'
model_path = 'models/comma_U_last.hdf5'
checkpoint_name = 'models/check.h5'
final_model = 'models/retrain.hdf5'

n_labels = num_classes()
total_loss, metrics = params(weights_path)
model = load_model(model_path, custom_objects = { 'dice_loss_plus_1focal_loss': total_loss,'jacard_coef': jacard_coef })

retrain = DataGenerator(data_dir=data_dir, shape=target_shape, split='retrain', batch_size=batch_size, shuffle=shuffle)

model.fit(retrain, epochs=epochs, verbose=1)
model.save(final_model)






    

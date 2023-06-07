import os
import numpy as np
from utils import params, DataGenerator, save_predictions, cat2rgb, jacard_coef, load_test_masks, evaluate
from keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_dir = 'data'
model_path = 'models/comma_U_mid.h5'
# pred_path = 'pred_Squeeze-Net'
weights_path = 'weights'
target_shape = (480, 288)
# output_shape = (288, 480, 3)
batch_size = 10

test = DataGenerator(data_dir=data_dir, shape =target_shape, batch_size=batch_size,split= 'test')

total_loss, metrics = params(weights_path)
    
model = load_model(model_path, custom_objects = { 'dice_loss_plus_1focal_loss': total_loss,'jacard_coef': jacard_coef })
pred = model.predict(test)

truth_argmax = load_test_masks(data_dir, target_shape)
pred = np.argmax(pred, axis=-1)
evaluate(truth_argmax, pred)

# pred = cat2rgb(pred, output_shape)
# save_predictions(pred, pred_path)

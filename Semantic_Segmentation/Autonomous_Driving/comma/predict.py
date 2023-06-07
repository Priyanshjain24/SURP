import os
from utils import params, jacard_coef, cat2rgb, save_predictions, DataGenerator
from keras.models import load_model
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_dir = 'All_Data/June-5'
weights_path = 'weights'
target_shape = (480, 288)
output_shape = (288, 480, 3)
model_path = 'models/comma_U_last.hdf5'

with open(os.path.join(data_dir, 'retrain/files'), 'r') as file:
    names = [line.strip() for line in file.readlines()]
total_loss, metrics = params(weights_path)
model = load_model(model_path, custom_objects = { 'dice_loss_plus_1focal_loss': total_loss,'jacard_coef': jacard_coef })
predict = DataGenerator(data_dir, target_shape, split='predict')

prediction = model.predict(predict)
prediction = cat2rgb(prediction, output_shape)

save_predictions(prediction, data_dir, names)


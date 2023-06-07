from keras.models import load_model
import numpy as np
import cv2
import segmentation_models as sm
from keras import backend as K

def predict(path):
    test_path = path

    def jacard_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

    weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
    dice_loss = sm.losses.DiceLoss(class_weights=weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    test = np.array(cv2.imread(test_path))
    height, width, channels = test.shape
    
    new = []
    new.append(cv2.resize(test,(256,256)))
    new = np.array(new)

    model = load_model('U-Net.hdf5', custom_objects = { 'dice_loss_plus_1focal_loss': total_loss,'jacard_coef': jacard_coef })
    pred = model.predict(new)
    predicted_img=np.argmax(pred, axis=3)[0,:,:]
    predicted_img = predicted_img.astype('uint8')
    predicted_img = cv2.resize(predicted_img, (width,height))

    return predicted_img

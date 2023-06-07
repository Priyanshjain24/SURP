from utils import *
from keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def test(data_dir, model_path, target_shape, output_shape):

    num_classes, test_images, test_masks = load_test_data(data_dir, target_shape)

    weights = weights_list(test_masks)
    dice_loss = sm.losses.DiceLoss(class_weights=weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss) 
    
    model = load_model(model_path, custom_objects = { 'dice_loss_plus_1focal_loss': total_loss,'jacard_coef': jacard_coef })
    pred = model.predict(test_images)
    pred = np.argmax(pred, axis = 3)
    pred = np.expand_dims(pred, axis = 3)
    pred = label2rgb(pred, num_classes, output_shape)

    return pred, test_masks

def main():

    data_dir = 'data'
    model_path = 'models/U_last.hdf5'
    pred_path = 'predictions_U'
    target_shape = (256, 128)
    output_shape = (128, 256, 3)

    prediction_masks, true_masks = test(data_dir, model_path, target_shape, output_shape)
    save_predictions(prediction_masks, pred_path)

if __name__ == "__main__":
    main()

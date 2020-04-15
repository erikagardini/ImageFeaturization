#References:
#[1] Lecoutre, A., Negrevergne, B., and Yger, F.
    # Rasta: Recognizing art style automatically in painting with deep learning.
        #In Asian Conference on Machine Learning, 2017.
#[2] https://github.com/bnegreve/rasta

from keras.models import load_model
from keras import backend as K
from keras.models import Model
import numpy as np
import os
from os.path import join
from progressbar import ProgressBar
from datetime import datetime
from keras.preprocessing.image import load_img,img_to_array
from utils.utils import imagenet_preprocess_input,get_dico,wp_preprocess_input,invert_dico
import csv


RESULT_FILE_PATH = '../models/results.csv'
MODEL_PATH = '../models/model.h5'
DATA_PATH = '../datasets/wikipaintings_full/wikipaintings_test'
RES_PATH = '../outputs'
k = [1, 3, 5]

def main():
    K.set_image_data_format('channels_last')


    preds = get_top_multi_acc(MODEL_PATH, DATA_PATH,top_k=k)
    for val,pred in zip(k,preds):
        print('\nTop-{} accuracy : {}%'.format(val,pred*100))

    model_name = "model_name"
    with open(RESULT_FILE_PATH,'a') as f:
           f.write('\n'+model_name+";"+str(preds[0])+";"+str(preds[1])+";"+str(preds[2])+';'+str(datetime.now()))

def get_top_multi_acc(model_path, test_data_path,top_k=[1,3,5]):
    y_pred, y = get_y_pred(model_path, test_data_path, max(top_k))
    scores = []
    for k in top_k:
        score = 0
        for pred, val in zip(y_pred[:,:k], y):
            if val in pred:
                score += 1
        scores.append(score / len(y))
    return scores

def get_y_pred(model_path, test_data_path,top_k=1):

    model = load_model(model_path)

    target_size =(224,224)

    dico = get_dico()
    y_true = []
    y_pred = []
    s = 0
    for t in list(os.walk(test_data_path)):
        s += len(t[2])
    style_names = os.listdir(test_data_path)
    if '.DS_Store' in style_names:
        style_names.remove('.DS_Store')
    print('Calculating predictions...')
    bar = ProgressBar(max_value=s)
    i = 0
    for style_name in style_names:
        style_path = join(test_data_path, style_name)
        img_names = os.listdir(style_path)
        if '.DS_Store' in img_names:
            img_names.remove('.DS_Store')
        label = dico.get(style_name)
        for img_name in img_names:
            img = load_img(join(style_path, img_name),target_size=target_size)
            x = img_to_array(img)

            #With bagging
            pred = _bagging_predict(style_name, img_name, x,model)

            args_sorted = np.argsort(pred)[0][::-1]
            y_true.append(label)
            y_pred.append([a for a in args_sorted[:top_k]])
            i += 1
            bar.update(i)
    return np.asarray(y_pred), y_true

def _bagging_predict(style_name, img_name, x,model):
    x_flip = np.copy(x)
    x_flip = np.fliplr(x_flip)
    x = imagenet_preprocess_input(x)
    x_flip = imagenet_preprocess_input(x_flip)
    pred = model.predict(x[np.newaxis,...])
    print(model.input)

    #mod_2
    _save_intermediate_output(model, x, style_name, img_name, layer=-2, filename=RES_PATH + '/img_dataset_testing.csv')

    pred_flip = model.predict(x_flip[np.newaxis,...])
    avg = np.mean(np.array([pred,pred_flip]), axis=0 )
    return avg

def _save_intermediate_output(model, x, style_name, img_name, layer=-2, filename=RES_PATH + '/img_dataset.csv'):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[layer].output)
    intermediate_output = intermediate_layer_model.predict(x[np.newaxis, ...])
    print(intermediate_output.shape)
    info_img = np.array([style_name, img_name])
    row = np.append(info_img, intermediate_output)
    with open(filename, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()

if __name__ == '__main__':

    main()

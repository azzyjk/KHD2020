import os
import argparse
import sys
import time
import numpy as np
import pandas as pd
from matplotlib.image import imread
import tensorflow as tf # Tensorflow 2
import arch
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM 
from sklearn.model_selection import train_test_split, KFold
import math
from tensorflow.keras.applications import *

######################## DONOTCHANGE ###########################
def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded!')

    def infer(image_path):
        result = []
        X = PathDataset(image_path, labels=None, batch_size = batch_size)
        y_hat = model.predict(X)            
        result.extend(np.argmax(y_hat, axis=1))

        print('predicted')
        return np.array(result)

    nsml.bind(save=save, load=load, infer=infer)


def path_loader (root_path):
    image_path = []
    image_keys = []
    for _,_,files in os.walk(os.path.join(root_path,'train_data')):
        for f in files:
            path = os.path.join(root_path,'train_data',f)
            if path.endswith('.png'):
                image_keys.append(int(f[:-4]))
                image_path.append(path)

    return np.array(image_keys), np.array(image_path)


def label_loader (root_path, keys):
    labels_dict = {}
    labels = []
    with open (os.path.join(root_path,'train_label'), 'rt') as f :
        for row in f:
            row = row.split()
            labels_dict[int(row[0])] = (int(row[1]))
    for key in keys:
        labels = [labels_dict[x] for x in keys]
    return labels
############################################################


class PathDataset(tf.keras.utils.Sequence): 
    def __init__(self,image_path, labels=None, batch_size=128, test_mode= True): 
        self.image_path = image_path
        self.labels = labels
        self.mode = test_mode
        self.batch_size = batch_size

    def __getitem__(self, idx): 
        image_paths = self.image_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([imread(x) for x in image_paths])
        
                ### REQUIRED: PREPROCESSING ###

        if self.mode:
            return batch_x
        else: 
            batch_y = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size])
            # return batch_x, batch_y
            out = list()
            for label in batch_y:
                if label == 1:
                    out.append([0,1])
                else :
                    out.append([1,0])
            out = np.array(out)
            return batch_x, out

    def __len__(self):
        return math.ceil(len(self.image_path) / self.batch_size)

if __name__ == '__main__':

    ########## ENVIRONMENT SETUP ############
    args = argparse.ArgumentParser()

    ########### DONOTCHANGE: They are reserved for nsml ###################
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    ######################################################################

    # hyperparameters
    args.add_argument('--epoch', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=32) 
    args.add_argument('--learning_rate', type=int, default=0.0003)

    config = args.parse_args()

    # training parameters
    num_epochs = config.epoch
    batch_size = config.batch_size
    num_classes = 2
    learning_rate = config.learning_rate  

    # model setting ## 반드시 이 위치에서 로드해야함
    # model = arch.cnn() 
    model = ResNet50V2(include_top=True, weights=None, input_shape=(512,512,3), classes=2, classifier_activation='softmax')

    # Loss and optimizer
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])


    ############ DONOTCHANGE ###############
    bind_model(model)
    if config.pause: ## test mode 일때는 여기만 접근
        print('Inferring Start...')
        nsml.paused(scope=locals())
    #######################################

    if config.mode == 'train': ### training mode 일때는 여기만 접근
        print('Training Start...')

        ############ DONOTCHANGE: Path loader ###############
        root_path = os.path.join(DATASET_PATH,'train')        
        image_keys, image_path = path_loader(root_path)
        labels = label_loader(root_path, image_keys)
        ##############################################
        
        dataList = list()

        for x, y in zip(labels, image_path):
            dataList.append([x, y])
            
        split = 5
        kf = KFold(n_splits = split, shuffle=True, random_state=42)
        data = pd.DataFrame(dataList, columns=['label', 'data'])
        

        for epoch in range(num_epochs):
            sumLoss = 0
            for train_index, test_index in kf.split(data):
                train_data, valid_data = data['data'][train_index], data['data'][test_index]
                train_label, valid_label = data['label'][train_index], data['label'][test_index]

                X = PathDataset(train_data, train_label, batch_size = batch_size, test_mode=False)
                validX = PathDataset(valid_data, valid_label, batch_size = batch_size, test_mode=False)

                hist = model.fit(X, validation_data=validX, verbose=2)
            nsml.report(summary=True, step=epoch, epoch_total=num_epochs, loss=sumLoss/5)#, acc=train_acc)
            nsml.save(epoch)

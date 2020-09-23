import os
import argparse
import numpy as np
from matplotlib.image import imread
import tensorflow as tf  # Tensorflow 2
from LuterGS import arch, Callbacks
import nsml
from nsml.constants import DATASET_PATH
import math
import random
import cv2


def setup_data(image_path, labels, validation_split=0.2):

    total = []
    for i in range(len(labels)):
        total.append([image_path[i], labels[i]])

    random.shuffle(total)
    x, y, val_x, val_y = [], [], [], []


    for i in range(len(labels)):
        if i < int(len(labels) * (1 - validation_split)):
            x.append(total[i][0])
            y.append(total[i][1])
        else:
            val_x.append(total[i][0])
            val_y.append(total[i][1])

    return x, y, val_x, val_y


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
        X = PathDataset(image_path, labels=None, batch_size=batch_size)
        y_hat = model.predict(X)
        result.extend(np.argmax(y_hat, axis=1))
        print('predicted')
        return np.array(result)

    nsml.bind(save=save, load=load, infer=infer)


def path_loader(root_path):
    image_path = []
    image_keys = []
    for _, _, files in os.walk(os.path.join(root_path, 'train_data')):
        for f in files:
            path = os.path.join(root_path, 'train_data', f)
            if path.endswith('.png'):
                image_keys.append(int(f[:-4]))
                image_path.append(path)

    return np.array(image_keys), np.array(image_path)


def label_loader(root_path, keys):
    labels_dict = {}
    labels = []
    with open(os.path.join(root_path, 'train_label'), 'rt') as f:
        for row in f:
            row = row.split()
            labels_dict[int(row[0])] = (int(row[1]))
    for key in keys:
        labels = [labels_dict[x] for x in keys]
    return labels


############################################################

def blurAndHsv(img, blur=True, hsv=True, gray=False):
    if blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if hsv:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


class PathDataset(tf.keras.utils.Sequence):
    def __init__(self, image_path, labels=None, batch_size=128, test_mode=True):
        self.image_path = image_path
        self.labels = labels
        self.mode = test_mode
        self.batch_size = batch_size

    def __getitem__(self, idx):
        image_paths = self.image_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([imread(x) for x in image_paths])
        batch_x_processed = []
        for img in batch_x:
            hsv = blurAndHsv(img, blur=True, hsv=False, gray=True)
            batch_x_processed.append(hsv)

        ### REQUIRED: PREPROCESSING ###

        if self.mode:
            # return batch_x
            return np.array(batch_x_processed)
        else:
            batch_y = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size])
            out = []
            for data in batch_y:
                if data == 1:
                    out.append([0, 1])
                else:
                    out.append([1, 0])

            out = np.array(out)

            # return batch_x, out
            return np.array(batch_x_processed), out

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
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--learning_rate', type=int, default=0.00001)

    config = args.parse_args()

    # training parameters
    num_epochs = config.epoch
    batch_size = config.batch_size
    num_classes = 2
    learning_rate = config.learning_rate

    # model setting ## 반드시 이 위치에서 로드해야함
    model = arch.cnn3(channel_num=1)

    # Loss and optimizer
    model.compile(tf.keras.optimizers.Adam(lr=0.00001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    ############ DONOTCHANGE ###############
    bind_model(model)
    if config.pause:  ## test mode 일때는 여기만 접근
        print('Inferring Start...')
        nsml.paused(scope=locals())
    #######################################

    if config.mode == 'train':  ### training mode 일때는 여기만 접근
        print('Training Start...')

        ############ DONOTCHANGE: Path loader ###############
        root_path = os.path.join(DATASET_PATH, 'train')
        image_keys, image_path = path_loader(root_path)
        labels = label_loader(root_path, image_keys)
        ##############################################

        print("this is path loader : ", image_path)
        print("this is labels:", labels)
        # image_path도 8000개, labels 도 8000개.
        # 이걸 이용해 validation_split도 만들 수 있지 않을까?

        for epoch in range(num_epochs):
            x, y, vx, vy = setup_data(image_path, labels, validation_split=0.35)
            train = PathDataset(x, y, batch_size=batch_size, test_mode=False)
            val = PathDataset(vx, vy, batch_size=batch_size, test_mode=False)

            hist = model.fit(train, validation_data=val, shuffle=True, verbose=1, callbacks=[Callbacks.lr_scheduler])
            nsml.report(summary=True, step=epoch, epoch_total=num_epochs, loss=hist.history['loss'])  # , acc=train_acc)
            nsml.save(epoch)

        # X = PathDataset(image_path, labels, batch_size=batch_size, test_mode=False)
        # X[0]은 epoch개수만큼의 x, y를 가지게 됨. 즉, X[0][0~39] 는 X의 테스트데이터, X[1][0~39]는 Y의 테스트데이터
        # X는 200까지 있음

        # x, y = setup_data(X, 40)

        # hist = model.fit(x=x, y=y, shuffle=True, epochs=50, batch_size=50, validation_split=0.25)
        # nsml.save()


        # for epoch in range(num_epochs):

            # hist = model.fit(X, shuffle=True)


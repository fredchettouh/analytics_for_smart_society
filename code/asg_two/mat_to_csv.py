#!/usr/bin/env python
# coding: utf-8


from scipy.io import loadmat
import pandas as pd
import numpy as np
import os

BASEDIR = os.getcwd()
DATA_DIR = 'data/asg_two'
DATA_FILE = 'HAR_database.mat'
META_DATA = 'database_description.txt'
COLNAMES_TRAIN = ['acc_z', 'acc_xy', 'gyro_x', 'gyro_y', 'gyro_z', 'label',
                  'id']
COLNAMES_TEST = ['acc_z', 'acc_xy', 'gyro_x', 'gyro_y', 'gyro_z', 'id']

TRAIN_FILE = 'HAR_train.csv'
TEST_FILE = 'HAR_test.csv'


def mat_train_to_csv():
    data = loadmat(os.path.join(BASEDIR, DATA_DIR, DATA_FILE))
    train_data = data['database_training']
    dataframes = []
    for i in range(train_data.shape[0]):
        temp = np.vstack((train_data[i][0], train_data[i][1]))
        ids = [i for meassurement in range(temp.shape[1])]
        temp = np.vstack((temp, ids))
        temp = np.transpose(temp)
        temp = pd.DataFrame((temp), columns=COLNAMES_TRAIN)
        dataframes.append(temp)
    return pd.concat(dataframes)


def mat_test_to_csv():
    data = loadmat(os.path.join(BASEDIR, DATA_DIR, DATA_FILE))
    test_data = data['database_test']
    dataframes = []
    for i in range(data['database_test'].shape[0]):
        temp = test_data[i][0]
        ids = [i for meassurement in range(temp.shape[1])]
        temp = np.vstack((temp, ids))
        temp = np.transpose(temp)
        temp = pd.DataFrame((temp), columns=COLNAMES_TEST)
        dataframes.append(temp)
    return pd.concat(dataframes)


if __name__ == "__main__":
    train = mat_train_to_csv()
    train.to_csv(os.path.join(BASEDIR, DATA_DIR, TRAIN_FILE), index=False)

    test = mat_test_to_csv()
    test.to_csv(os.path.join(BASEDIR, DATA_DIR, TEST_FILE), index=False)

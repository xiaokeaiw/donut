import numpy as np
import pandas as pd
from donut import complete_timestamp, standardize_kpi
import sys
import os
import time

# Read the raw data.
# file = os.path.join('./data/AIOPS2018/',sys.argv[1])
# df = pd.read_csv(file)
# timestamp, values, labels = df['timestamp'],df['value'],df['label']
file_list = os.listdir(sys.argv[1])

timestamp = np.array([])
values = np.array([])
labels = np.array([])
missing = np.array([])

for file in file_list:
    df_now = pd.read_csv(os.path.join(sys.argv[1],file))
    timestamp_now, values_now, labels_now = df_now['timestamp'],df_now['value'],df_now['label']
    timestamp_now, missing_now, (values_now, labels_now) = \
    complete_timestamp(timestamp_now, (values_now, labels_now))
    values_now = values_now.astype(float)
    missing2_now = np.isnan(values_now)
    values_now[np.where(missing2_now==1)[0]] = 0
    missing_now = np.logical_or(missing_now,missing2_now)

    timestamp = np.append(timestamp,timestamp_now)
    values = np.append(values,values_now)
    labels = np.append(labels,labels_now)
    missing = np.append(missing,missing_now)


# If there is no label, simply use all zeros.
#labels = np.zeros_like(values, dtype=np.int32)

# Complete the timestamp, and obtain the missing point indicators.
# timestamp, missing, (values, labels) = \
#     complete_timestamp(timestamp, (values, labels))

# values = values.astype(float)
# missing2 = np.isnan(values)
# values[np.where(missing2==1)[0]] = 0
# missing = np.logical_or(missing,missing2)

# Split the training and testing data.
test_portion = 0.5
test_n = int(len(values) * test_portion)
train_values, test_values = values[:-test_n], values[-test_n:]
train_labels, test_labels = labels[:-test_n], labels[-test_n:]
train_missing, test_missing = missing[:-test_n], missing[-test_n:]

# Standardize the training and testing data.
train_values, mean, std = standardize_kpi(
    train_values, excludes=np.logical_or(train_labels, train_missing))
test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

import tensorflow as tf
from donut import Donut
from tensorflow import keras as K
from tfsnippet.modules import Sequential

# We build the entire model within the scope of `model_vs`,
# it should hold exactly all the variables of `model`, including
# the variables created by Keras layers.
with tf.variable_scope('model') as model_vs:
    model = Donut(
        h_for_p_x=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        h_for_q_z=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        x_dims=120,
        z_dims=5,
    )

from donut import DonutTrainer, DonutPredictor

trainer = DonutTrainer(model=model, model_vs=model_vs)
predictor = DonutPredictor(model)


def point_adjust(score, label , thres):
    predict = score >= thres
    actual = label > 0.1
    anomaly_state = False
    
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict, actual

def calc_p2p(predict, actual):
    tp = np.sum(predict * actual)
    tn = np.sum((1-predict) * (1-actual))
    fp = np.sum(predict * (1-actual))
    fn = np.sum((1-predict) * actual)
    
    precision = (tp+0.000001) / (tp + fp + 0.000001)
    recall = (tp+0.000001) / (tp + fn + 0.000001)
    f1 = (2 * precision * recall)/ (precision + recall )
    #print(tp,fp,fn,f1)
    return f1, precision, recall, tp, tn, fp, fn

def delay_point_adjust(score, label, thres,delay=7):

    predict = score >= thres
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


with tf.Session().as_default():
    trainer.fit(train_values, train_labels, train_missing, mean, std)
    time1 = time.time()
    test_score = -predictor.get_score(test_values, test_missing)
    time2 = time.time()

    print('test_time',time2-time1)
    print(len(test_score))
    print(len(test_labels))
    max_th = float(max(test_score))
    min_th = float(min(test_score))
    grain = 2000
    max_f1 = 0.0
    max_f1_th = 0.0
    max_pre = 0.0
    max_recall = 0.0
    delay_f1 = 0.0
    delay_f1_th = 0.0
    delay_pre = 0.0
    delay_recall = 0.0
    label = test_labels[119:]
    for i in range(0,grain+3):
        thres = (max_th-min_th)/grain*i+min_th
        predict, actual = point_adjust(test_score, label, thres=thres)
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_th = thres
            max_pre = precision
            max_recall = recall
    print(max_f1,max_pre,max_recall)
    for i in range(0,grain+3):
        thres = (max_th-min_th)/grain*i+min_th
        predict = delay_point_adjust(test_score,label,thres,7)
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, label)
        if f1 > delay_f1:
            delay_f1 = f1
            delay_f1_th = thres
            delay_pre = precision
            delay_recall = recall
    print(delay_f1,delay_pre,delay_recall)
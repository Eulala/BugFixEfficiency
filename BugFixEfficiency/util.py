import math
import json
import os
import tqdm
import pickle
from datetime import datetime
import pandas as pd
import time
import numpy
import csv
from sklearn.cluster import KMeans
import configparser
from myMongo import *

Global_val = { 'mongo_config': {'ip': '', 'port': 0, 'username': '', 'pwd': '', 'db_name': ''},
               'data_dir': '', 'figure_dir': '', 'result_dir': '',
               'commit_dir': ''}


def initialize():
    config = load_config()
    global Global_val
    set_global_val('data_dir', config['DataPath']['root_dir'])
    # Global_val['data_dir'] = config['DataPath']['root_dir']
    if not os.path.exists(Global_val['data_dir']):
        os.mkdir(Global_val['data_dir'])

    set_global_val('figure_dir', config['DataPath']['figure_dir'])
    if not os.path.exists(Global_val['figure_dir']):
        os.mkdir(Global_val['figure_dir'])

    set_global_val('result_dir', config['DataPath']['result_dir'])
    if not os.path.exists(Global_val['result_dir']):
        os.mkdir(Global_val['result_dir'])

    set_global_val('commit_dir', config['DataPath']['commit_dir'])
    if not os.path.exists(Global_val['commit_dir']):
        raise ValueError('no such commit dir')

    mongo = config['MongoDB']
    set_global_val('mongo_config', {'ip': mongo['ip'], 'port': int(mongo['port']), 'username': mongo['username'],
                                    'pwd': mongo['pwd'], 'db_name': mongo['db_name']})


def get_global_val(key):
    return Global_val[key]


def set_global_val(key, data):
    global Global_val
    Global_val[key] = data


def write_json_list(data, filename):
    start = time.time()
    with open(filename, 'w') as f:
        for i in data:
            f.write(json.dumps(i)+'\n')
    end = time.time()
    print('write {} lines to {} runtime: {}'.format(len(data), filename, end-start))


def write_json_dict(data, filename):
    start = time.time()
    with open(filename, 'w') as f:
        for i in data:
            f.write(json.dumps({'_id': i, 'data': data[i]})+'\n')
    end = time.time()
    print('write {} lines to {} runtime: {}'.format(len(data), filename, end-start))


def write_json_data(data, filename):
    start = time.time()
    with open(filename, 'w') as f:
        f.write(json.dumps(data))
    end = time.time()
    print('write {} lines to {} runtime: {}'.format(len(data), filename, end-start))


def load_json_list(filename):
    start = time.time()
    data = []
    with open(filename, 'r') as f:
        for i in f:
            dic = json.loads(i)
            data.append(dic)
    end = time.time()
    print('load {} lines from {} runtime: {}'.format(len(data), filename, end-start))
    return data


def load_json_dict(filename):
    start = time.time()
    data = {}
    with open(filename, 'r') as f:
        for i in f:
            dic = json.loads(i)
            data[dic['_id']] = dic['data']
    end = time.time()
    print('load {} lines from {} runtime: {}'.format(len(data), filename, end-start))
    return data


def load_json_data(filename):
    start = time.time()
    data = {}
    with open(filename, 'r') as f:
        for i in f:
            data = json.loads(i)
    end = time.time()
    print('load {} lines from {} runtime: {}'.format(len(data), filename, end-start))
    return data


def load_from_disk(filename):
    start = time.time()
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    end = time.time()
    print('load from {} runtime: {}'.format(filename, end-start))
    return obj


def calculate_delta_t(time1, time2, unit='d'):
    format = '%Y-%m-%dT%H:%M:%SZ'
    a = datetime.strptime(time1, format)
    b = datetime.strptime(time2, format)
    t1 = time.mktime(a.timetuple()) * 1000 + a.microsecond / 1000
    t2 = time.mktime(b.timetuple()) * 1000 + b.microsecond / 1000
    a = abs(t2 - t1)
    b = a / 1000 / 3600  # hour
    c = int(b / 24)  # day

    if unit == 'h':  # hour
        return math.ceil(b)
    elif unit == 'd':  # day
        return math.ceil(c)
    elif unit == 'm':  # minute
        return math.ceil(a/1000/60)
    elif unit == 's':
        return math.ceil(a / 1000)


def func_none():
    print("cannot find func")


def normalize(data):
    res = []
    _max = numpy.max(data)
    _min = numpy.min(data)
    for d in data:
        td = (d-_min)/(_max-_min)
        res.append(td)
    return res


def delete_outlier(data, index):
    qs = numpy.percentile(data, (25, 50, 75), method='midpoint')
    iqr = qs[2]-qs[0]
    outlier = qs[2] + 3*iqr

    for i in index:
        index[i] = data[index[i]]
    res = []
    new_index = {}
    count = 0
    for i in index:
        if index[i] <= outlier:
            res.append(index[i])
            new_index[i] = count
            count = count + 1

    return res, new_index


# def generate_event_id(events, write_path):
#     event_id = dict(zip(events, range(len(events))))
#
#     # mapping the alphabet
#     for e in event_id:
#         if event_id[e] < 26:
#             event_id[e] = chr(ord('A')+event_id[e])
#         else:
#             event_id[e] = chr(ord('a')+event_id[e]-26)
#
#     write_json_data(event_id, write_path)


def create_config():
    c_file = configparser.ConfigParser()
    c_file.add_section("MongoDB")
    c_file.set("MongoDB", "mongo_ip", "172.27.135.32")
    c_file.set("MongoDB", "port", "27017")
    c_file.set("MongoDB", "username", "oss")
    c_file.set("MongoDB", "pwd", "oss")
    c_file.set("MongoDB", "db_name", "oss_behavior")

    with open(r'configurations.ini', 'w') as f:
        c_file.write(f)
        f.flush()
        f.close()


def load_config():
    config = configparser.ConfigParser()
    config.read('configurations.ini')
    return config


def calcu_prefix_sum_b(data):
    if len(data) == 0:
        return [], [], []
    pos = [0] * len(data)
    neg = [0] * len(data)
    neu = [0] * len(data)
    for i in range(len(data)):
        if data[i][1] == 'pos':
            pos[i] = 1
        elif data[i][1] == 'neg':
            neg[i] = 1
        else:
            neu[i] = 1
    sum_p = [0] * len(data)
    sum_n = [0] * len(data)
    sum_u = [0] * len(data)
    sum_p[0] = pos[0]
    sum_n[0] = neg[0]
    sum_u[0] = neu[0]
    for i in range(1, len(pos)):
        sum_p[i] = sum_p[i - 1] + pos[i]
        sum_n[i] = sum_n[i - 1] + neg[i]
        sum_u[i] = sum_u[i-1] + neu[i]
    return sum_p, sum_n, sum_u


def calculate_entropy_b(p_num, n_num, u_num, total):
    H_L = 0
    H_R = 0
    H_U = 0
    if p_num != 0:
        H_L = - (p_num / total) * numpy.log2(p_num / total)
    if n_num != 0:
        H_R = - (n_num / total) * numpy.log2(n_num / total)
    if u_num != 0:
        H_U = - (u_num / total) * numpy.log2(u_num / total)
    H_0 = H_L + H_R + H_U
    return H_0

def calculate_label_num_b(sum_p, sum_n, sum_u, split_pos):
    # start = time.time()
    label_num = [0]*6

    last_p = 0
    last_n = 0
    if split_pos == 0:
        label_num[0] = 0
        label_num[1] = 0
        label_num[2] = 0
    else:
        label_num[0] = sum_p[split_pos-1]
        label_num[1] = sum_n[split_pos-1]
        label_num[2] = sum_u[split_pos-1]

    label_num[3] = sum_p[len(sum_p)-1] - label_num[0]
    label_num[4] = sum_n[len(sum_n)-1] - label_num[1]
    label_num[5] = sum_u[len(sum_u) - 1] - label_num[2]
    # end = time.time()
    # print(end-start)
    return label_num

def find_split_by_entropy_b(data, sum_p=[], sum_n=[], sum_u=[]):
    if len(data) == 0:
        return -1

    _min = data[0][0]
    _max = data[len(data) - 1][0]
    if _min == _max:
        return -1
    label_num = [0, 0, 0, 0, 0, 0]

    if len(sum_p) == len(sum_n) == 0:
        sum_p, sum_n, sum_u = calcu_prefix_sum_b(data)
    p_num = sum_p[len(sum_p)-1]
    n_num = sum_n[len(sum_n)-1]
    u_num = sum_u[len(sum_u)-1]
    label_num[3] = p_num
    label_num[4] = n_num
    label_num[5] = u_num
    # print(label_num)
    if label_num[3] == len(data) or label_num[4] == len(data) or label_num[5] == len(data):  # already divided into 2 classes
        return -1

    H_0 = calculate_entropy_b(p_num, n_num, u_num, len(data))
    IG = calculate_information_gain_b(label_num, len(data), H_0)
    best_split = 0
    max_I = IG
    # print(IG)
    i = 0
    while len(data)-1 > i > -1:
        split_i = get_split_b(data[i+1:len(data)], data[i][0])
        if split_i < 0:
            i = len(data)
        else:
            i = split_i + (i + 1)
        label_num = calculate_label_num_b(sum_p, sum_n, sum_u, i)
        # print(data[j][0], data[k][0], data[m][0])
        # print(i, label_num)
        IG = calculate_information_gain_b(label_num, len(data), H_0)
        # print(IG)
        # print(max_I)
        if IG > max_I:
            max_I = IG
            best_split = i
        # print(i, len(data), IG)
    # print(max_I)
    return best_split


def calculate_information_gain_b(label_num, data_num, H):
    # print(label_num)
    _bin = int(len(label_num)/3)
    calcu_part = [0]*len(label_num)
    _num = [0]*_bin
    for i in range(len(label_num)):
        k = math.floor(i / 3)
        _num[k] += label_num[i]
    # print(_num)
    for i in range(len(label_num)):
        if label_num[i] != 0:
            k = math.floor(i / 3)
            calcu_part[i] = - (label_num[i] / _num[k]) * (numpy.log2(label_num[i] / _num[k]))

    # print(calcu_part)
    LH = [0]*_bin
    for i in range(len(label_num)):
        k = math.floor(i / 3)
        LH[k] += calcu_part[i]

    IG = H
    for i in range(_bin):
        IG -= LH[i] * (_num[i]/data_num)
    return IG

def get_split_b(data, ini_val):
    for i in range(len(data)):
        if data[i][0] > ini_val:
            return i
    return -1
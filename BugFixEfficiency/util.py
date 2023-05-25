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
               'data_dir': '', 'figure_dir': '',
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


def calculate_delta_t(time1, time2):
    format = '%Y-%m-%dT%H:%M:%SZ'
    a = datetime.strptime(time1, format)
    b = datetime.strptime(time2, format)
    t1 = time.mktime(a.timetuple()) * 1000 + a.microsecond / 1000
    t2 = time.mktime(b.timetuple()) * 1000 + b.microsecond / 1000
    a = t2 - t1
    b = a / 1000 / 3600  # hour
    c = int(b / 24)  # day


    return math.ceil(c)


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

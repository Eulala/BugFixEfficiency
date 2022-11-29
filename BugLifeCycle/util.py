import math
import pymongo
import json
from datetime import datetime
import pandas as pd
import time
import numpy
import csv
from sklearn.cluster import KMeans


def write_json_data(data, path):
    with open(path, 'w') as f:
        for i in data:
            f.write(json.dumps(i)+'\n')


def load_json_data(path):
    data = []
    with open(path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            data.append(dic)
    return data


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

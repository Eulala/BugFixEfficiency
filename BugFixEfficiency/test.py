import csv
import json
import os

import numpy
import numpy as np
from util import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pattern.text.en import sentiment
from preprocess import *
from contrast_sequential_pattern_mining import *
from sequence_cluster import *
from random import sample
from main import *
bots = {'tensorflowbutler', 'google-ml-butler', 'tensorflow-bot', 'copybara-service', 'tensorflow-copybara', 'ansible', 'ansibot'}


def draw_violin_plot(data_path, save_path):
    df = pd.read_csv(data_path)
    # print(df)
    f, ax = plt.subplots(figsize=(11, 6))

    sns.violinplot(data=df, palette="Set3", bw=.2, cut=1, linewidth=1, showmeans=True)

    # sns.despine(left=True, bottom=True)
    plt.ylabel('length')
    plt.savefig(save_path, dpi=150)


if __name__ == '__main__':

    initialize()
    data_dir = get_global_val('result_dir') + 'entropy_test/'
    CDSPM(data_dir)


    exit(-1)


    # classify_issues()
    # model_sequence()
    # data_dir = get_global_val('data_dir')+'sequences/'
    #
    # length = 15
    # cut_len = length*2+1
    # files = os.listdir(data_dir)
    # files = list(filter(lambda x: '_model' in x, files))
    # for file in files:
    #     res = []
    #     data = load_json_list(os.path.join(data_dir, file))
    #     for d in data:
    #         temp = {'_id': d['_id'], 'action_sequence': d['action_sequence'][0:cut_len]}
    #         res.append(temp)
    #
    #     file = file.replace("model", 'cut_origin')
    #     write_json_list(res, os.path.join(data_dir, file))


    # qs = numpy.percentile(res, (25, 50, 75), method='midpoint')
    # Q3 = qs[2]
    # IQR = qs[2]-qs[0]
    # K = Q3 + 1.5*IQR
    # print(K)

    # df = pd.DataFrame(res)
    # f, ax = plt.subplots(figsize=(11, 6))
    #
    # bp = sns.boxplot(data=df, palette="Set3", linewidth=1, showmeans=True)
    #
    # # sns.despine(left=True, bottom=True)
    # plt.ylabel('fix-time')
    # plt.savefig(figure_dir+'tensorflow_issue_fix_time', dpi=150)

    # print(len(fix_time))
    # res = []
    # for i in fix_time:
    #     temp = []
    #     for j in fix_time[i]:
    #         temp.append(fix_time[i][j])
    #     print(len(temp))
    #     max_time = max(temp)
    #     min_time = min(temp)
    #     print(i, max_time, min_time)
    #     res.append([i, min_time, max_time])
    #
    #
    #
    # df = pd.DataFrame(res, columns=['year', 'min', 'max'])
    # print(df)

    exit(-1)




    # data = load_json_list(os.path.join(data_dir, 'issue_sequences_tensorflow_neg_cut_origin.json'))
    # for i in data:
    #     for a in range(len(i['action_sequence'])):
    #         try:
    #             if i['action_sequence'][a] == 'H' and i['action_sequence'][a+1] > 99967 and i['action_sequence'][a+2] == 'H':
    #                 print(i)
    #                 break
    #         except Exception:
    #             continue

    # data = load_json_data(os.path.join(data_dir, 'tensorflow_issues_fix_time.json'))
    # res = {}
    # for i in data:
    #     if len(data[i]['seq']) > 15:
    #         res[i] = data[i]
    # write_json_dict(res, os.path.join(data_dir, 'tensorflow_issues_len15_fix_time.json'))
    # exit(-1)

    # classify:
    # data = load_json_dict(os.path.join(data_dir, 'tensorflow_issues_len15_fix_time.json'))
    # fix_time = []
    # for i in data:
    #     fix_time.append(data[i]['fix_time'])
    # qs = numpy.percentile(fix_time, (25, 50, 75), method='midpoint')
    # print(qs)
    # fast = {}
    # slow = {}
    # for i in data:
    #     if data[i]['fix_time'] <= qs[0]:
    #         fast[i] = data[i]
    #     elif data[i]['fix_time'] >= qs[2]:
    #         slow[i] = data[i]
    # write_json_dict(fast, os.path.join(data_dir, 'tensorflow_issues_len15_fast.json'))
    # write_json_dict(slow, os.path.join(data_dir, 'tensorflow_issues_len15_slow.json'))

    fast = load_json_dict(os.path.join(data_dir, 'tensorflow_issues_len15_fast.json'))
    slow = load_json_dict(os.path.join(data_dir, 'tensorflow_issues_len15_slow.json'))
    res = {'neg': [], 'pos': []}
    for i in slow:
        level = 'neg'
        temp = {'_id': i, 'action_sequence': []}
        for a in slow[i]['seq']:
            # if a['actor'] not in bots:
            temp['action_sequence'].append({'event_type': a['event_type'], 'occur_at': a['occur_at']})
        res[level].append(temp)
    for i in fast:
        level = 'pos'
        temp = {'_id': i, 'action_sequence': []}
        for a in fast[i]['seq']:
            # if a['actor'] not in bots:
            temp['action_sequence'].append({'event_type': a['event_type'], 'occur_at': a['occur_at']})
        res[level].append(temp)

    for level in ['neg', 'pos']:
        write_json_list(res[level], data_dir + 'sequences/issue_sequences_tensorflow_' + level + '.json')

    exit(-1)





    # print(len(res), qs)
    # df = pd.DataFrame(res, columns=['ratio'])
    #
    # figure_dir = get_global_val('figure_dir')
    # draw_boxplot(df, figure_dir+'tensorflow_last_close_ratio', '')
    # write_json_list(res, os.path.join(data_dir, 'tensorflow_resolved_issues_closebyself.json'))

    # data = load_json_data(os.path.join(data_dir, 'issue_comments_sorted.json'))
    # for r in data:
    #     if r != 'tensorflow':
    #         continue
    #     d = data[r]['55270']
    #     for c in d:
    #         print(c['comment'], sentiment(c['comment']))
        # for n in data[r]:
        #     for c in data[r][n]:
        #         if 'thank' in c['comment']:
        #             print(r, n, c['comment'], sentiment(c['comment']))
        #             exit(-1)

    data = load_json_list(data_dir+'tensorflow_stalled_issues.json')
    exist = set()
    for i in data:
        _id = i['repo_name'] + '_' + str(i['target']['number'])
        exist.add(_id)

    data = load_json_list(data_dir+'preprocessed_closed_issue_discussion.json')
    res = []
    others = []
    for i in data:
        _id = i['repo_name'] + '_' + str(i['target']['number'])
        if _id in exist:
            continue
        is_reopen = False
        close_actor = ''
        for e in i['action_sequence']:
            if e['event_type'] == 'ReopenedEvent':
                is_reopen = True
            if e['event_type'] == 'ClosedEvent':
                close_actor = e['actor']
        if not is_reopen and close_actor not in bots:
            res.append(i)
        else:
            others.append(i)
    write_json_list(res, data_dir+'tensorflow_resolved_issues_2.json')
    write_json_list(others, data_dir + 'tensorflow_failed_issues_2.json')







    # last_event = {}
    # last_event_2 = {}
    # events = set()
    # count = 0
    # for i in data:
    #     a_len = len(i['action_sequence'])
    #     e = i['action_sequence'][a_len-2]['event_type']
    #     events.add(e)
    #     if e not in last_event:
    #         last_event[e] = 0
    #     last_event[e] += 1
    #     if a_len >= 10:
    #         if e not in last_event_2:
    #             last_event_2[e] = 0
    #         last_event_2[e] += 1
    #         count += 1
    # print(last_event, last_event_2)
    #
    # event_id = dict(zip(events, range(len(events))))
    #
    # # mapping the alphabet
    # for e in event_id:
    #     if event_id[e] < 26:
    #         event_id[e] = chr(ord('A') + event_id[e])
    #     else:
    #         event_id[e] = chr(ord('a') + event_id[e] - 26)
    #
    # res = []
    # for i in last_event:
    #     res.append([event_id[i], last_event[i], 'all'])
    # for i in last_event_2:
    #     res.append([event_id[i], last_event_2[i], 'length >= 10'])
    #
    # df = pd.DataFrame(res, columns=['event', 'occurrence', 'type'])
    # df = df.sort_values(by=['occurrence'], ascending=[False])
    # print(df, count)
    # print(event_id)
    #




    # data = load_json_list(data_dir + 'issue_fix_time.json')
    # res = []
    #
    # key = set()
    # data2 = load_json_list(data_dir+'preprocessed_closed_issue_discussion_21.json')
    # for i in data2:
    #     _id = i['repo_name'].split('/')[1] + '_' + str(i['target']['number'])
    #     key.add(_id)
    # for i in data:
    #     if i['_id'] in key:
    #         repo = i['_id'].split('_')[0]
    #         res.append([repo, i['data']])
    #
    # df = pd.DataFrame(res, columns=['repo', 'close_time'])
    #
    # figure_dir = get_global_val('figure_dir')
    # draw_histplot(df, figure_dir+'issue close time', 'issue close time(days)')





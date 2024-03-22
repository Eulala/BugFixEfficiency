import csv
import json
import math
import os

import numpy
import numpy as np
from util import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import *
from contrast_sequential_pattern_mining import *
from sequence_cluster import *
from random import sample
from main import *
bots = {'tensorflowbutler', 'google-ml-butler', 'tensorflow-bot', 'copybara-service', 'tensorflow-copybara', 'ansible', 'ansibot', 'github-project-automation', 'pytorchmergebot'}


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
    data_dir = get_global_val('result_dir')

    # for repo in ['tensorflow', 'go', 'rust', 'transformers', 'angular',
    #              'flutter', 'rails', 'vscode', 'kubernetes',  'node',
    #              'godot', 'react-native', 'fastlane', 'electron', 'core',
    #              'pytorch']:
    for repo in ['total']:
        # 'core'
        min_len = 9
        for max_len in range(10, 30):
            data_dir = os.path.join(get_global_val('result_dir'),
                                    "{}_{}_new".format(repo, str(min_len), str(max_len)))
            total_test_3 = []
            total_pred_3 = []
            for count in range(1, 11):
                test, pred, temp_list = validate_seq_vector_3(data_dir, count, min_sup=0.05, min_gr=1.5,
                                                              use_csp=False, use_PCA=False)
                total_test_3 += test
                for i in pred:
                    total_pred_3.append(i)

            print('------------fast vs median vs slow-------------')
            print(confusion_matrix(total_test_3, total_pred_3, labels=['pos', 'neu', 'neg']))
            print(classification_report(total_test_3, total_pred_3))
            write_json_data(classification_report(total_test_3, total_pred_3, output_dict=True),
                            os.path.join(data_dir, 'classification_report_3_fsp.json'))
    exit(-1)


    # for max_len in range(10, 31):
    #     data_dir = get_global_val('result_dir') + repo + '_' + str(min_len-1) + '_' + str(max_len)
    #
    #     false_seq = []
    #     total_test = []
    #     total_pred = []
    #     for count in range(1, 11):
    #         test, pred, temp_seq = validate_seq_vector(data_dir, count, use_csp=False)
    #         false_seq += temp_seq
    #         total_test += test
    #         for i in pred:
    #             total_pred.append(i)
    #     print(confusion_matrix(total_test, total_pred, labels=['pos', 'neg']))
    #     print(classification_report(total_test, total_pred))
    #     print(false_seq)
    #     write_json_data(classification_report(total_test, total_pred, output_dict=True),
    #                     os.path.join(data_dir, 'fsp_classification_report.json'))

    # for gr in range(10, 31):
    #     gr = gr/10
    #     data_dir = get_global_val('result_dir') + 'tensorflow_9_30_total_gr_'+str(gr)
    #
    #     X, Y, D = generate_dataset(repo_name)
    #     if not os.path.exists(data_dir):
    #         os.mkdir(data_dir)
    #     dataset_time_discretize(X, Y, data_dir)
    #     generate_input_sequence_ete(X, Y, data_dir, 'input_sequences_0.json', use_entropy=True)
    #     CDSPM(data_dir, 0, min_gr=gr)

        # X, Y, D = generate_dataset(repo_name)
        # if not os.path.exists(data_dir):
        #     os.mkdir(data_dir)
        # write_json_list(D, os.path.join(data_dir, 'all_sequences.json'))
        #
        # # d = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=10)
        # d = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
        #
        # total_test = []
        # total_pred = []
        # count = 1
        # for train_idx, test_idx in d.split(X, Y):
        #     x_train, x_test = numpy.array(X, dtype=object)[train_idx], numpy.array(X, dtype=object)[test_idx]
        #     y_train, y_test = numpy.array(Y, dtype=object)[train_idx], numpy.array(Y, dtype=object)[test_idx]
        #     dataset_time_discretize(x_train, y_train, data_dir)
        #     write_json_list([train_idx.tolist(), test_idx.tolist()], os.path.join(data_dir, 'split_index.json'))
        #
        #     generate_input_sequence_ete(x_train, y_train, data_dir, 'input_sequences_' + str(count) + '.json',
        #                                 use_entropy=True)
        #     generate_input_sequence_ete(x_test, y_test, data_dir, 'test_sequences_' + str(count) + '.json',
        #                                 use_entropy=True)
        #     # generate_all_sequence_ete(D, data_dir, 'all_sequences_symbol_ver.json', use_entropy=True)
        #
        #     CDSPM(data_dir, count, min_gr=gr)
        #     test, pred, temp_list = validate_seq_vector(data_dir, count)
        #     total_test += test
        #     for i in pred:
        #         total_pred.append(i)
        #
        #     count += 1
        #     # a = validate_seq(data_dir)
        #     # acc.append(a)
        #     # exit(-1)
        # # print(numpy.array(acc).mean())
        #
        # print(confusion_matrix(total_test, total_pred, labels=['pos', 'neg']))
        # print(classification_report(total_test, total_pred))
        # write_json_data(classification_report(total_test, total_pred, output_dict=True),
        #                 os.path.join(data_dir, 'classification_report.json'))

    exit(-1)
    # model_idx = 10
    # data_dir = os.path.join(get_global_val('result_dir'), 'ansible_9_30')
    # test_ = load_json_dict(os.path.join(data_dir, 'test_sequences_' + str(model_idx) + '.json'))
    # test_seqs = {'pos': [], 'neg': []}
    # test_seqs_neg = []
    # count = 0
    # for i in test_:
    #     for q in test_[i]:
    #         test_seqs[i].append({'seq': q})
    #         count += 1
    #         # if count == 267:
    #         #     print(q)
    #
    # for i in test_seqs:
    #     for j in test_seqs[i]:
    #         j['cut_seq'] = j['seq'][0:9]
    #
    # pos_s = ''
    # neg_s = ''
    # for i in test_seqs['pos']:
    #     for j in test_seqs['neg']:
    #         if i['cut_seq'] == j['cut_seq']:
    #             print(i['cut_seq'])
    #             print(i['seq'])
    #             print(j['seq'])
    #             pos_s = i['seq']
    #             neg_s = j['seq']
    #
    #             patterns = []
    #             data = load_json_data(os.path.join(data_dir, 'pos_0.1_sup_csp_' + str(model_idx) + '.json'))
    #             for m in data:
    #                 patterns.append(m['seq'])
    #             data = load_json_data(os.path.join(data_dir, 'neg_0.1_sup_csp_' + str(model_idx) + '.json'))
    #             for m in data:
    #                 patterns.append(m['seq'])
    #             cand = set()
    #             for m in patterns:
    #                 for n in m:
    #                     cand.add(n)
    #
    #             # # seq = ['R*X', 'X+E', 'E-H', 'H+Q', 'Q+E', 'E-H', 'H-F', 'F+E', 'E+E', "E-X", "X+U", "U-H", "H+H", "H-H"]
    #             clf = pickle.load(open(os.path.join(data_dir, 'model_' + str(model_idx) + '.sav'), 'rb'))
    #             # pred = clf.predict(seq)
    #             for s in [pos_s, neg_s]:
    #                 print('------------------------------------------------------------')
    #                 print('The full sequence: {}'.format(s))
    #                 new_s = s[0:9]
    #                 subsequent_events = s[9:len(s)]
    #                 do_recommend(new_s, patterns, clf, cand)
    #                 for k in subsequent_events:
    #                     new_s = new_s + [k]
    #                     do_recommend(new_s, patterns, clf, cand)



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





import json
import numpy as np
from util import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import *
from contrast_sequential_pattern_mining import *


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
    select_closed_issue()
    exit(-1)
    # with open(r'data/closed_bug_fix.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         loc = dic['LOC']
    #         temp = 0
    #         for l in loc:
    #             temp = temp + l['add'] + l['del']
    #         if 1000000 < temp < 1200000:
    #             print(temp)
    #             print(dic)
    #             break


    # df = pd.read_csv('data/clusters_features_2.csv')
    # df = pd.read_csv('data/clusters_features_comparison_2.csv')
    # # print(df.loc[(df['loc'] > 10000) & (df['loc'] < 100000) & (df['repo_name'] == 'ansible')].count())
    #
    #
    # # temp = df.loc[df['loc_difference_ratio'] > 0.75]
    # df.loc[:, 'loc_difference_ratio'] = df.loc[:, 'loc_difference'] / df.loc[:, 'old_loc']
    # # print(df.loc[(df['loc_difference_ratio'] == 0) & (df['repo_name'] == 'ansible')].count())
    #
    # # print(df.loc[:, 'old_loc'])
    # f, ax = plt.subplots(figsize=(11, 6))
    #
    # # ax.set(ylim=(0, 600))
    # # temp = df[['old_loc', 'new_loc']]
    # # temp_df = temp[df['repo_name'].isin(['ansible'])]
    # # sns.boxplot(data=temp_df, palette="Set3", showmeans=True)
    # # sns.violinplot(data=temp, palette="Set3", bw=.2, cut=1, linewidth=1, showmeans=True)
    # # plt.show()
    # temp = df[(df['repo_name'] == 'ansible')]
    # print(temp)
    # sns.displot(data=temp['loc_difference_ratio'])
    # sns.despine(left=True, bottom=True)
    # plt.title('ansible')
    # plt.savefig('figures/loc_difference_ratio.png', dpi=150)

    # df = pd.read_csv('data/clusters_features.csv')
    # print(df.loc[(df['cluster'] == 2) & (df['efficiency_level'] == 'high'), 'loc'].count())
    # print(df.loc[(df['repo_name'] == 'ansible') & (df['cluster'] == 2), 'loc'].median())

    # with open('data/commit_diffs_limited.json', 'r') as f:
    #     count = 0
    #     for i in f:
    #         dic = json.loads(i)
    #         if dic['data']['add'] + dic['data']['del'] > 10000:
    #             print(dic['sha'], dic['data']['add'] + dic['data']['del'], dic['data']['msg'])
    #             print('')
    #             count = count + 1
    # print(count)

    # print(df.loc[(df['repo_name'] == 'ansible') & (df['loc'] < 100000) & (df['loc'] > 10000), :])
    # with open(r'data/closed_bug_fix.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         if dic['number'] == 53740:
    #             print(dic)
    #             print(dic['linked_pr'], len(dic['linked_commit']))
    #             for j in range(len(dic['linked_commit'])):
    #                 print(dic['linked_commit'][j], dic['LOC'][j])
    # with open(r'data/closed_bug_fix.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         total = 0
    #         for l in dic['LOC']:
    #             total = total + l['add']+l['del']
    #         if total > 10000:
    #             print(dic['repo_name'], dic['number'], total)
    # with open('temp_data/test.json', 'r') as f:
    #     dic = json.load(f)
    #     print(len(dic['files']))
    #     for file in dic['files']:
    #         if file['additions']+file['deletions'] > 500:
    #             print(file)
    # temp = set()
    # with open(r'data/closed_bug_fix.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         if '133395a44e16ced8a9e524f1d1208b690681feab' in dic['linked_commit']:
    #             print(dic['number'])
    #             temp.add(dic['number'])
    # print(len(temp))
    # { 'event_type': 'CrossReferencedEvent', 'created_at': '2016-09-09T21:38:24Z', 'actor': 'skg-net' }
    # with open(r'data/bug_issues_with_resolutions.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         if dic['number'] == 39390:
    #             print(dic)

    # with open(r'data/pr_events.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         if dic['number'] == 44337:
    #             print(dic)


    # commits = set()
    # with open('data/can_not_found_commit.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         commits.add(dic)
    #
    # commit_repo = {}
    # with open('data/issue_events.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         if dic['data']['__typename'] == 'ReferencedEvent' and 'commit' in dic['data']:
    #             commit_repo[dic['data']['commit']['oid']] = dic['data']['commitRepository']['nameWithOwner']
    # with open('data/pr_events.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         if dic['data']['__typename'] == 'PullRequestCommit' and 'commit' in dic['data']:
    #             commit_repo[dic['data']['commit']['oid']] = dic['data']['resourcePath'].split('/')[1]+'/'+dic['data']['resourcePath'].split('/')[2]
    #             # print(commit_repo[dic['data']['commit']['oid']])
    #             # exit(-1)
    # res = []
    #
    # for c in commits:
    #     url = "https://api.github.com/repos/"+commit_repo[c]+"/commits/"+c+"?per_page=100"
    #     res.append(url)
    #
    # with open(r'data/commitDetailUrls.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['_id'])
    #     for l in res:
    #         writer.writerow([l])




    # draw_violin_plot('data/sequence_num.csv', 'figures/sequence_num.png')
    # res = []
    # dic = {}
    # with open(r'data/closed_bug_fix_clusters_efficiency.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         temp = dic['3']
    #         for j in temp:
    #             res.append(j['efficiency'])
    # e_mean = numpy.mean(res)
    # low = []
    # high = []
    #
    # issue_event_num = {}
    # with open(r'data/closed_bug_fix.json', 'r') as f:
    #     for i in f:
    #         temp = json.loads(i)
    #         issue_event_num[temp['number']] = len(temp['events'])+1
    #         if issue_event_num[temp['number']] > 300:
    #             print(temp['number'])
    #
    # for i in dic['3']:
    #     if i['efficiency'] < e_mean:
    #         low.append(issue_event_num[int(i['number'])])
    #     else:
    #         high.append(issue_event_num[int(i['number'])])
    #
    # df = pd.concat([pd.DataFrame({'low_efficiency': low}), pd.DataFrame({'high_efficiency': high})], axis=1)
    # df.to_csv('temp.csv', index=False)
    # draw_violin_plot('temp.csv', 'figures/cluster_4_length.png')

    # res = []
    # efficient = []
    # with open('data/closed_bug_issue_classes.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         # res.append(dic['features']['fix_efficiency'])
    #         if dic['features']['fix_efficiency'] > 432:
    #             res.append(dic['issue_id'])
    #         else:
    #             efficient.append(dic['issue_id'])
    # # print(np.percentile(res, (25, 50, 75), method='midpoint'))
    #
    # issue_label = {}
    # with open('data/tensorflow_issue_label.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         if dic['number'] not in issue_label:
    #             issue_label[dic['number']] = []
    #             issue_label[dic['number']].append(dic['name'])
    #
    #



    # issue_event_num = {}
    # with open(r'data/closed_bug_fix.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         issue_event_num[dic['number']] = len(dic['events'])+1
    #
    # res = {'0': [], '1': [], '2': [], '3': []}
    # with open(r'data/tensorflow_body_cluster.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         for num in dic:
    #             res[dic[num]].append(issue_event_num[int(num)])
    #
    # max_len = numpy.max([len(res['0']), len(res['1']), len(res['2']), len(res['3'])])
    # with open(r'data/sequence_num.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3'])
    #     for i in range(max_len):
    #         temp = [None, None, None, None]
    #         for u in res:
    #             if i < len(res[u]):
    #                 temp[int(u)] = res[u][i]
    #         writer.writerow(temp)

    # res = {'0': [], '1': [], '2': [], '3': []}
    # with open(r'data/closed_bug_fix_clusters_efficiency.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         for j in dic:
    #             for u in dic[j]:
    #                 res[j].append(u['efficiency'])
    #
    # # print(res)
    # for i in res:
    #     e_mean = numpy.mean(res[i])
    #     low = []
    #     high = []
    #     for j in res[i]:
    #         if j < e_mean:
    #             low.append(j)
    #         else:
    #             high.append(j)
    #     llow_1 = []
    #     llow_2 = []
    #     hhigh_1 = []
    #     hhigh_2 = []
    #     e_mean = numpy.mean(low)
    #     for j in low:
    #         if j < e_mean:
    #             llow_1.append(j)
    #         else:
    #             llow_2.append(j)
    #     e_mean = numpy.mean(high)
    #     for j in high:
    #         if j < e_mean:
    #             hhigh_1.append(j)
    #         else:
    #             hhigh_2.append(j)
    #     print(len(llow_1), len(llow_2), len(hhigh_1), len(hhigh_2))

    # event_id = load_json_data(r'data/event_id.json')[0]
    #
    # trans_event = {}
    # for e in event_id:
    #     trans_event[event_id[e]] = e
    # # print(trans_event)
    #
    # head = ''
    # res = []
    # with open(r'data/2CSP_results.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         if row[0] == 'CSP':
    #             head = row
    #             continue
    #         sequence = ''
    #         for i in row[0]:
    #             e = trans_event[i]
    #             sequence += e + '-'
    #         row[0] = sequence
    #         res.append(row)
    #
    # with open(r'data/2CSP_with_name.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(head)
    #     for i in res:
    #         writer.writerow(i)

    # df = pd.read_csv('data/0CSP_with_name.csv')
    # temp = df.loc[('Assigne' in df['CSP']) & ('Unassign' in df['CSP'])]
    # print(temp)
    # # print(df.loc[(df['Sup_1']+df['Sup_2'] > 50) & (df['Class'] == 1)].count())
    # count = 0
    # with open(r'data/2CSP_with_name.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         if row[0] == 'CSP':
    #             continue
    #         n = row[0].split('-')
    #         if len(n) >= 11:
    #             print(row[0], row[6])
    #             count += 1
    # print(count)

    # df = pd.read_csv('result/repo_metric_coefficient.csv')
    # temp = df.loc[(df['repo'] == 'total')]
    # matrix = np.zeros((15, 15))
    # metric_map = { }
    # metric_map_reverse = {}
    # count = 0
    # for j in temp:
    #     col_name = j
    #     if col_name == 'repo':
    #         continue
    #     r = col_name.split('-')[0]
    #     c = col_name.split('-')[1]
    #
    #     if r not in metric_map:
    #         metric_map[r] = count
    #         metric_map_reverse[count] = r
    #         count += 1
    #     if c not in metric_map:
    #         metric_map[c] = count
    #         metric_map_reverse[count] = c
    #         count += 1
    #
    #     n_r = metric_map[r]
    #     n_c = metric_map[c]
    #     value = temp[j].tolist()[0]
    #     matrix[n_r, n_c] = value
    #
    # head = ['']
    # for i in range(15):
    #     head.append(metric_map_reverse[i])
    # with open('result/total_coef_matrix.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(head)
    #     for r in range(15):
    #         temp_list = [metric_map_reverse[r]]
    #         for j in range(15):
    #             temp_list.append(matrix[r, j])
    #         writer.writerow(temp_list)
    # level_matrix = [[] for i in range(15)]
    # print(level_matrix)
    # for i in range(15):
    #     for j in range(15):
    #         r = matrix[i, j]
    #         level = matrix[i, j]
    #         if abs(r) <= 1:
    #             if abs(r) == 1:
    #                 level = 'P'
    #             elif abs(r) >= 0.7:
    #                 level = 'S'
    #             elif abs(r) >= 0.4:
    #                 level = 'M'
    #             elif abs(r) >= 0.1:
    #                 level = 'W'
    #             else:
    #                 level = 'N'
    #         level_matrix[i].append(level)
    # with open('result/total_coef_level_matrix.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(head)
    #     for r in range(15):
    #         temp_list = [metric_map_reverse[r]]
    #         for j in range(15):
    #             temp_list.append(level_matrix[r][j])
    #         writer.writerow(temp_list)
    #
    # df = pd.read_csv('data/clusters_features.csv')
    # temp_df = df.loc[(df['cluster'] == 2), ['fix_efficiency', 'fix_time']]
    #
    # print(temp_df.corr())

    df = pd.read_csv('data/2CSP_results.csv')
    print(df.loc[(df['Class'] == 1)].count())
    print(df.loc[(df['Class'] == 2)].count())

    # with open('data/closed_bug_fix_sequences.json', 'r') as f:
    #     dic = json.load(f)
    #     low = dic['1']['low']
    #     high = dic['1']['high']
    #     low_list = []
    #     high_list = []
    #     for d in low:
    #         low_list.append(d['sequence'])
    #     for d in high:
    #         high_list.append(d['sequence'])
    #
    #
    #     print(len(low_list), len(high_list))
    #     count_1 = 0
    #     count_2 = 0
    #     target = 'EKEX'
    #     for source in low_list:
    #         flag = False
    #         if target == source:
    #             count_1 += 1
    #             continue
    #         cur_pos_t = 0
    #         cur_pos_s = 0
    #         while cur_pos_s < len(source):
    #             if target[cur_pos_t] == source[cur_pos_s]:
    #                 cur_pos_t = cur_pos_t + 1
    #             cur_pos_s = cur_pos_s + 1
    #             if cur_pos_t == len(target):
    #                 flag = True
    #                 break
    #
    #         if flag is True:
    #             # print(source)
    #             # target is a subsequence of source
    #             count_1 += 1
    #
    #     for source in high_list:
    #         flag = False
    #         if target == source:
    #             count_2 += 1
    #             continue
    #         cur_pos_t = 0
    #         cur_pos_s = 0
    #         while cur_pos_s < len(source):
    #             if target[cur_pos_t] == source[cur_pos_s]:
    #                 cur_pos_t = cur_pos_t + 1
    #             cur_pos_s = cur_pos_s + 1
    #             if cur_pos_t == len(target):
    #                 flag = True
    #                 break
    #
    #         if flag is True:
    #             # target is a subsequence of source
    #             count_2 += 1
    #
    #     print(count_1, count_2)
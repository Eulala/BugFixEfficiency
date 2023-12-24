import math
import os.path

from util import *
from bs4 import BeautifulSoup
import nltk
from output import *

# state_map = {'new': 1, 'comprehended': 2, 'assigned': 3, 'proposed': 4, 'passed': 5, 'closed': 6, 'failed': 7, 'discussed': 8}
bots = {'tensorflowbutler', 'google-ml-butler', 'tensorflow-bot', 'copybara-service', 'tensorflow-copybara',
        'ansible', 'ansibot', 'github-project-automation', 'pytorchmergebot', 'gopherbot', 'ngbot', 'github-actions',
        'VSCodeTriageBot', }


def extract_raw_data():
    mongo_config = get_global_val('mongo_config')
    data_dir = get_global_val('data_dir')
    mongo_c = MyMongo(mongo_config['ip'], mongo_config['username'], mongo_config['pwd'], port=int(mongo_config['port']))
    mongo_c.set_db_name(mongo_config['db_name'])
    mongo_c.connect()

    data = mongo_c.get_col_value(col_name='issue_discussion', cond={'repo_name': {"$in": ['nodejs/node']}, 'behavior_type': 'collective'})
    write_json_list(data, data_dir + 'node_issue_discussion.json')
    mongo_c.close()

    # mongo_c = MyMongo(mongo_config['ip'], 'sbh', 'sbh123456', port=int(mongo_config['port']))
    # mongo_c.set_db_name('ghdb')
    # mongo_c.connect()
    # data = mongo_c.get_col_value(col_name='issueComment', cond={"index.repo_name": {"$in": ['tensorflow', 'godot']}})
    # write_json_list(data, data_dir+'issue_comments.json')

    # data = mongo_c.get_col_value(col_name='pullRequestTimeline', cond={"index.repo_name": {"$in": ['tensorflow', 'ansible']}})
    # write_json_list(data, data_dir+'pr_events.json')
    # mongo_c.close()


def issue_preprocess(repo_name):
    data_dir = get_global_val('data_dir')
    issues = load_json_list(os.path.join(data_dir, repo_name+'_issue_discussion.json'))
    # ignore_events = {'SubscribedEvent', 'UnsubscribedEvent', 'LockedEvent', 'UnlockedEvent'}
    ignore_events = set()
    open_issues = []
    closed_issues = []
    for i in issues:
        if i['action_sequence'][0]['actor'] in bots:
            continue
        temp_act = []
        is_close = False
        for a in i['action_sequence']:
            if a['event_type'] not in ignore_events:
                temp_act.append(a)
            if a['event_type'] == 'ClosedEvent':
                is_close = True
        # if not is_close:
        #     continue

        del_index = []
        for j in range(len(temp_act)):
            a = temp_act[j]
            if a['event_type'] == 'MentionedEvent':
                try:
                    b = temp_act[j+1]
                    if b['event_type'] == 'SubscribedEvent':
                        # MentionedEvent and SubscribedEvent of the same time compose an @ event
                        del_index.append(j+1)
                except Exception:
                    pass
            elif a['event_type'] == 'IssueComment':
                for u in range(j+1, len(temp_act)):
                    if temp_act[u]['event_type'] == 'MentionedEvent':
                        # @ more than one user
                        if calculate_delta_t(a['occur_at'], temp_act[u]['occur_at'], unit='s') < 2:
                            del_index.append(u)
        del_index = list(set(del_index))
        del_index.sort(reverse=True)
        try:
            for j in del_index:
                temp_act.pop(j)
        except Exception:
            print(del_index, temp_act)
            exit(-1)
        if is_close:
            # from the start to the last closed event
            for j in range(len(temp_act)-1, 0, -1):
                if temp_act[j]['event_type'] == 'ClosedEvent':
                    i['action_sequence'] = temp_act[0:j+1]
                    break
        if is_close:
            closed_issues.append(i)
        else:
            open_issues.append(i)
    write_json_list(closed_issues, os.path.join(data_dir, 'preprocessed_'+repo_name+'_closed_issue_discussion.json'))
    write_json_list(open_issues, os.path.join(data_dir, 'preprocessed_'+repo_name+'_open_issue_discussion.json'))


def delete_closed_by_bot(repo_name):
    data_dir = get_global_val('data_dir')
    data = load_json_list(os.path.join(data_dir, 'preprocessed_'+repo_name+'_closed_issue_discussion.json'))

    res = []
    for i in data:
        seq_len = len(i['action_sequence'])
        if i['action_sequence'][seq_len-1]['actor'] in bots:
            continue
        res.append(i)

    write_json_list(res, os.path.join(data_dir, repo_name+'_closed_issues.json'))


def select_issue_longer_than(min_len, repo_name):
    data_dir = get_global_val('data_dir')
    data = load_json_list(os.path.join(data_dir, repo_name+'_closed_issues.json'))

    res = []
    for i in data:
        seq_len = len(i['action_sequence'])
        if seq_len > min_len:
            _id = i['repo_name'].split('/')[1]+'_'+str(i['target']['number'])
            temp_dict = {'_id': _id, 'action_sequence': i['action_sequence']}
            res.append(temp_dict)

    write_json_list(res, os.path.join(data_dir, repo_name+'_closed_issues_len'+str(min_len)+'.json'))


def calculate_fix_time(repo_name, min_len):
    data_dir = get_global_val('data_dir')
    filename = repo_name+'_closed_issues_len'+str(min_len)
    data = load_json_list(os.path.join(data_dir, filename+'.json'))

    res = {}
    for i in data:
        t1 = i['action_sequence'][0]['occur_at']
        t2 = i['action_sequence'][len(i['action_sequence'])-1]['occur_at']
        delta_t = calculate_delta_t(t1, t2, unit='m')
        res[i['_id']] = delta_t

    write_json_data(res, os.path.join(data_dir, filename + '_fix_time.json'))


def calculate_avg_res_time(repo_name, min_len):
    data_dir = get_global_val('data_dir')
    filename = repo_name+'_closed_issues_len'+str(min_len)
    data = load_json_list(os.path.join(data_dir, filename+'.json'))

    res = {}
    for i in data:
        t1 = i['action_sequence'][0]['occur_at']
        t2 = i['action_sequence'][len(i['action_sequence'])-1]['occur_at']
        delta_t = calculate_delta_t(t1, t2, unit='m')
        seq_len = len(i['action_sequence'])
        res[i['_id']] = delta_t/(seq_len-1)

    write_json_data(res, os.path.join(data_dir, filename + '_avg_time.json'))


def calculate_person_time(repo_name, min_len):
    data_dir = get_global_val('data_dir')
    filename = repo_name+'_closed_issues_len'+str(min_len)
    data = load_json_list(os.path.join(data_dir, filename+'.json'))

    res = {}
    for i in data:
        t1 = i['action_sequence'][0]['occur_at']
        t2 = i['action_sequence'][len(i['action_sequence'])-1]['occur_at']
        delta_t = calculate_delta_t(t1, t2, unit='m')
        actors = set()
        for e in i['action_sequence']:
            actors.add(e['actor'])

        res[i['_id']] = delta_t/(len(actors))

    write_json_data(res, os.path.join(data_dir, filename + '_person_time.json'))


def classify_sequence(repo_name, len_, use_fix=True):
    data_dir = get_global_val('data_dir')
    if repo_name == 'total':
        files = list(filter(lambda x: '_closed_issues_len'+str(len_)+'.json' in x and 'ansible' not in x, os.listdir(data_dir)))
        data = []
        for f in files:
            data += load_json_list(os.path.join(data_dir, f))
        files = list(filter(lambda x: '_closed_issues_len' + str(len_) in x and 'fix_' in x and 'ansible' not in x, os.listdir(data_dir)))
        fix_time = {}
        for f in files:
            temp_dict = load_json_data(os.path.join(data_dir, f))
            fix_time.update(temp_dict)
    else:
        data = load_json_list(os.path.join(data_dir, repo_name+'_closed_issues_len'+str(len_)+'.json'))
        if use_fix:
            fix_time = load_json_data(os.path.join(data_dir, repo_name+'_closed_issues_len'+str(len_)+'_fix_time.json'))
        else:
            fix_time = load_json_data(
                os.path.join(data_dir, repo_name + '_closed_issues_len' + str(len_) + '_avg_time.json'))

    qs = numpy.percentile(list(fix_time.values()), (25, 50, 75), method='midpoint')
    Q1 = qs[0]
    Q3 = qs[2]
    IQR = qs[2]-qs[0]
    K_max = Q3 + 1.5*IQR
    K_min = Q1 - 1.5*IQR

    print(qs, K_max, K_min)

    year_fix_time = {}
    for i in data:
        t = fix_time[i['_id']]
        if t > K_max or t < K_min:
            # outlier
            continue

        year = i['action_sequence'][0]['occur_at'].split('-')[0]
        if year not in year_fix_time:
            year_fix_time[year] = []
        year_fix_time[year].append(t)

    max_fix_time = {}
    min_fix_time = {}
    for y in year_fix_time:
        max_fix_time[y] = max(year_fix_time[y])
        min_fix_time[y] = min(year_fix_time[y])

    nor_fix_time = {}
    for i in data:
        t = fix_time[i['_id']]
        if t > K_max or t < K_min:
            # outlier
            continue
        year = i['action_sequence'][0]['occur_at'].split('-')[0]
        try:
            nor_t = (t-min_fix_time[year])/(max_fix_time[year]-min_fix_time[year])
        except Exception:
            nor_t = 1
        nor_fix_time[i['_id']] = nor_t

    slow_i = []
    fast_i = []
    median_i = []

    fix_time_ = list(fix_time.values())
    new_fix_time = []
    for i in fix_time_:
        if i >= 1*24*60:
            new_fix_time.append(i)
    qs_2 = numpy.percentile(new_fix_time, (25, 50, 75), method='midpoint')
    print(qs_2)

    # min~thresholds: fast, thresholds~max: slow
    # thresholds = 90*60*24
    # min_time = 30*60*24
    # qs = qs_2
    # thresholds = qs[1]
    # print(thresholds)
    for i in data:
        try:
            # if fix_time[i['_id']] <= 1*24*60:
            #     continue
            if fix_time[i['_id']] <= 3*24*60:
                fast_i.append(i)
            elif fix_time[i['_id']] > 3*24*60:
                slow_i.append(i)
            else:
                median_i.append(i)
        except Exception:
            pass
    #
    # for i in data:
    #     try:
    #         if len(i['action_sequence']) <= 25:
    #             fast_i.append(i)
    #         elif len(i['action_sequence']) > 25:
    #             slow_i.append(i)
    #         else:
    #             median_i.append(i)
    #     except Exception:
    #         pass

    qs = numpy.percentile(list(nor_fix_time.values()), (25, 50, 75), method='midpoint')
    # qs = numpy.percentile(list(nor_fix_time.values()), (40, 50, 60), method='midpoint')
    # min~q1: fast, q3~max: slow

    # max_fast = 0
    # for i in data:
    #     try:
    #         if nor_fix_time[i['_id']] <= qs[0]:
    #             fast_i.append(i)
    #             if fix_time[i['_id']] > max_fast:
    #                 max_fast = fix_time[i['_id']]
    #         elif nor_fix_time[i['_id']] >= qs[2]:
    #             slow_i.append(i)
    #         else:
    #             median_i.append(i)
    #     except Exception:
    #         pass
    #
    # print(max_fast)
    thresholds = 999999999

    write_json_list(slow_i, os.path.join(data_dir, repo_name+'_issues_len'+str(len_)+'_slow.json'))
    write_json_list(fast_i, os.path.join(data_dir, repo_name+'_issues_len'+str(len_)+'_fast.json'))
    write_json_list(median_i, os.path.join(data_dir, repo_name + '_issues_len' + str(len_) + '_median.json'))

    # delete actor and the last closed event
    pos = []
    neg = []
    med = []
    for i in slow_i:
        temp = {'_id': i['_id'], 'action_sequence': []}
        for a in range(len(i['action_sequence'])-1):
            e = i['action_sequence'][a]
            temp['action_sequence'].append({'event_type': e['event_type'], 'occur_at': e['occur_at']})
        neg.append(temp)
    for i in fast_i:
        temp = {'_id': i['_id'], 'action_sequence': []}
        for a in range(len(i['action_sequence']) - 1):
            e = i['action_sequence'][a]
            temp['action_sequence'].append({'event_type': e['event_type'], 'occur_at': e['occur_at']})
        pos.append(temp)
    for i in median_i:
        temp = {'_id': i['_id'], 'action_sequence': []}
        for a in range(len(i['action_sequence']) - 1):
            e = i['action_sequence'][a]
            temp['action_sequence'].append({'event_type': e['event_type'], 'occur_at': e['occur_at']})
        med.append(temp)

    data_dir = os.path.join(data_dir, repo_name)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    write_json_list(pos, os.path.join(data_dir, 'issue_sequences_pos.json'))
    write_json_list(neg, os.path.join(data_dir, 'issue_sequences_neg.json'))
    write_json_list(med, os.path.join(data_dir, 'issue_sequences_med.json'))
    return thresholds
    # return max_fast

def select_stalled_issues():
    data_dir = get_global_val('data_dir')
    data = load_json_list(data_dir+'preprocessed_closed_issue_discussion.json')

    res = []
    for i in data:
        _id = i['repo_name'].split('/')[1] + '_' + str(i['target']['number'])
        for e in range(len(i['action_sequence'])):
            d = i['action_sequence'][e]
            if d['actor'] == 'google-ml-butler' and d['event_type'] == 'LabeledEvent':
                if d['supple_data']['label_name'] == 'stalled':
                    try:
                        m = i['action_sequence'][e+1]
                        n = i['action_sequence'][e+2]
                        if m['actor'] == 'google-ml-butler' and m['event_type'] == 'IssueComment' and n['actor'] == 'google-ml-butler' and n['event_type'] == 'ClosedEvent':
                            res.append(i)
                            break
                    except Exception:
                        continue
    write_json_list(res, data_dir+'tensorflow_stalled_issues.json')











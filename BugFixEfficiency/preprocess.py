import math
import os.path

from util import *
from bs4 import BeautifulSoup
import nltk
from output import *

# state_map = {'new': 1, 'comprehended': 2, 'assigned': 3, 'proposed': 4, 'passed': 5, 'closed': 6, 'failed': 7, 'discussed': 8}
bots = {'tensorflowbutler', 'google-ml-butler', 'tensorflow-bot', 'copybara-service', 'tensorflow-copybara', 'ansible', 'ansibot'}


def extract_raw_data():
    mongo_config = get_global_val('mongo_config')
    data_dir = get_global_val('data_dir')
    # mongo_c = MyMongo(mongo_config['ip'], mongo_config['username'], mongo_config['pwd'], port=int(mongo_config['port']))
    # mongo_c.set_db_name(mongo_config['db_name'])
    # mongo_c.connect()
    #
    # data = mongo_c.get_col_value(col_name='issue_discussion', cond={'repo_name': {"$in": ['godotengine/godot']}, 'behavior_type': 'collective'})
    # write_json_list(data, data_dir + 'godot_issue_discussion.json')
    # mongo_c.close()

    mongo_c = MyMongo(mongo_config['ip'], 'sbh', 'sbh123456', port=int(mongo_config['port']))
    mongo_c.set_db_name('ghdb')
    mongo_c.connect()
    data = mongo_c.get_col_value(col_name='issueComment', cond={"index.repo_name": {"$in": ['tensorflow', 'godot']}})
    write_json_list(data, data_dir+'issue_comments.json')

    # data = mongo_c.get_col_value(col_name='pullRequestTimeline', cond={"index.repo_name": {"$in": ['tensorflow', 'ansible']}})
    # write_json_list(data, data_dir+'pr_events.json')
    # mongo_c.close()


def find_commit_repo():
    data_dir = get_global_val('data_dir')
    res = {}
    issue_events = load_json_list(data_dir+'issue_events.json')
    for i in issue_events:
        if i['data']['__typename'] == "ReferencedEvent":
            try:
                oid = i['data']['commit']['oid']
                repo = i['data']['commitRepository']['nameWithOwner']
                res[oid] = repo
            except Exception:
                pass

    pr_events = load_json_list(data_dir+'pr_events.json')
    for i in pr_events:
        if i['data']['__typename'] == "PullRequestCommit":
            try:
                oid = i['data']['commit']['oid']
                repo = i['index']['repo_owner']+'/' + i['index']['repo_name']
                res[oid] = repo
            except Exception:
                pass

    write_json_dict(res, data_dir+'commit_repos.json')


def get_commit_list():
    data_dir = get_global_val('data_dir')
    if os.path.exists(data_dir+'commit_list.json'):
        commits = load_json_list(data_dir+'commit_list.json')
        commits = set(commits)
        return commits
    else:
        commits = set()
        data = load_json_list(data_dir + 'bug_fix.json')
        for i in data:
            events = i['action_sequence']
            for e in events:
                if e['event_type'] == 'ReferencedEvent':
                    commits.add(e['supple_data']['oid'])
                elif e['event_type'] == 'PullRequestEvent':
                    for sub_e in e['sub_event']:
                        if sub_e['event_type'] == 'PullRequestCommit':
                            commits.add(sub_e['supple_data']['oid'])
        write_json_list(commits, data_dir+'commit_list.json')
        return commits


def select_commits(commit_list):
    commit_dir = get_global_val('commit_dir')
    data_dir = get_global_val('data_dir')
    res = {}
    files = os.listdir(commit_dir)
    files = list(filter(lambda f: 'enrich' not in f, files))
    for f in files:
        commits = load_from_disk(commit_dir+f)
        for c in tqdm.tqdm(commits):
            if c in commit_list:
                res[c] = commits[c]

    write_json_dict(res, data_dir+'commits.json')


def normalize_fix_time():
    data_dir = get_global_val('data_dir')
    data = load_json_list(data_dir + 'bug_fix_time.json')

    temp = {}
    for i in data:
        if i['repo'] not in temp:
            temp[i['repo']] = []
        temp[i['repo']].append(i['fix_time'])

    _max = {'tensorflow/tensorflow': max(temp['tensorflow/tensorflow']), 'ansible/ansible': max(temp['ansible/ansible'])}
    _min = {'tensorflow/tensorflow': min(temp['tensorflow/tensorflow']), 'ansible/ansible': min(temp['ansible/ansible'])}

    res = []
    for i in data:
        i['fix_time'] = (i['fix_time'] - _min[i['repo']])/(_max[i['repo']] - _min[i['repo']])
        res.append(i)

    write_json_list(res, data_dir+'bug_fix_time_nor.json')


def set_efficiency():
    data_dir = get_global_val('data_dir')
    data = load_json_list(data_dir + 'bug_fix_time_nor.json')
    res = []
    for repo in ['ansible/ansible', 'tensorflow/tensorflow']:
        f_data = list(filter(lambda d: d['repo'] == repo, data))
        f_data = sorted(f_data, key=lambda x: x['fix_time'])
        median_k = math.ceil(len(f_data)/2) - 1

        for i in range(len(f_data)):
            if i < median_k:
                f_data[i]['efficiency'] = 'high'
            elif i > median_k:
                f_data[i]['efficiency'] = 'low'
            res.append(f_data[i])

    write_json_list(res, data_dir+'bug_fix_with_efficiency.json')


def issue_preprocess():
    data_dir = get_global_val('data_dir')
    issues = load_json_list(data_dir + 'tensorflow_issue_discussion.json')
    # ignore_events = {'SubscribedEvent', 'UnsubscribedEvent', 'LockedEvent', 'UnlockedEvent'}
    ignore_events = set()
    open_issues = []
    closed_issues = []
    for i in issues:
        if i['action_sequence'][0]['actor'] in bots:
            continue
        elif i['repo_name'] not in ['tensorflow/tensorflow']:
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
                        del_index.append(j+1)
                except Exception:
                    pass
            elif a['event_type'] == 'IssueComment':
                for u in range(j+1, len(temp_act)):
                    if temp_act[u]['event_type'] == 'MentionedEvent':
                        if calculate_delta_t(a['occur_at'], temp_act[u]['occur_at'], unit='s') < 2:
                            del_index.append(u)
        del_index.sort(reverse=True)
        # del_index.reverse()
        for j in del_index:
            temp_act.pop(j)
        if is_close:
            for j in range(len(temp_act)-1, 0, -1):
                if temp_act[j]['event_type'] == 'ClosedEvent':
                    i['action_sequence'] = temp_act[0:j+1]
                    break
        # print(i['repo_name'], i['target']['number'])
        # mention_time = set()
        # for a in i['action_sequence']:
        #     if a['event_type'] == 'MentionedEvent':
        #         mention_time.add(a['occur_at'])
        # try:
        #     # print(mention_time)
        #     del_index = []
        #     index = 0
        #     subs_time = set()
        #     for a in i['action_sequence']:
        #         subs_time.add(a['occur_at'])
        #         if a['event_type'] == 'SubscribedEvent' and a['occur_at'] in mention_time:
        #             del_index.append(index)
        #         index += 1
        #     del_index.reverse()
        #     # print(subs_time)
        #     # print(del_index)
        #     for j in del_index:
        #         i['action_sequence'].pop(j)
        # except Exception:
        #     pass  # median
        if is_close:
            closed_issues.append(i)
        else:
            open_issues.append(i)
    write_json_list(closed_issues, data_dir + 'preprocessed_closed_issue_discussion.json')
    write_json_list(open_issues, data_dir + 'preprocessed_open_issue_discussion.json')


def cat_comments_to_issues():
    data_dir = get_global_val('data_dir')
    issues = load_json_list(os.path.join(data_dir, 'preprocessed_closed_issue_discussion.json'))
    if not os.path.exists(os.path.join(data_dir, 'issue_comments_sorted.json')):
        comments = load_json_list(os.path.join(data_dir, 'issue_comments.json'))
        issue_comment = {}
        for i in tqdm.tqdm(comments):
            repo = i['index']['repo_name']
            number = i['index']['number']
            if repo not in issue_comment:
                issue_comment[repo] = {}
            if number not in issue_comment[repo]:
                issue_comment[repo][number] = []
            try:
                temp = {'occur_at': i['data']['createdAt'], 'actor': None,
                        'comment': i['data']['bodyHTML']}
                if 'author' in i['data']:
                    temp['actor'] = i['data']['author']['login']
                temp['comment'] = html_document_str(temp['comment'])
                issue_comment[repo][number].append(temp)
            except Exception:
                print(i)
                exit(-1)
        write_json_data(issue_comment, os.path.join(data_dir, 'issue_comments_sorted.json'))
    comments = load_json_data(os.path.join(data_dir, 'issue_comments_sorted.json'))

    for i in tqdm.tqdm(issues):
        repo = i['repo_name'].split('/')[1]
        number = str(i['target']['number'])
        try:
            comment = comments[repo][number]
        except Exception:
            continue
        for e in range(len(i['action_sequence'])):
            if i['action_sequence'][e]['event_type'] == 'IssueComment':
                try:
                    for c in comment:
                        if c['occur_at'] == i['action_sequence'][e]['occur_at'] and c['actor'] == i['action_sequence'][e]['actor']:
                            i['action_sequence'][e]['supple_data'] = c['comment']
                            break
                except Exception:
                    pass
    write_json_list(issues, os.path.join(data_dir, 'preprocessed_closed_issue_discussion_with_comments.json'))


def delete_less_than(length=20):
    data_dir = get_global_val('data_dir')
    issues = load_json_list(data_dir + 'preprocessed_closed_issue_discussion.json')
    res = []
    for i in issues:
        if len(i['action_sequence']) < length:
            continue
        else:
            res.append(i)
    write_json_list(res, data_dir + 'preprocessed_closed_issue_discussion_'+str(length)+'.json')


def delete_faster_than(_time=1):
    data_dir = get_global_val('data_dir')
    issues = load_json_list(data_dir + 'preprocessed_closed_issue_discussion.json')
    res = []
    for i in issues:
        t0 = i['action_sequence'][0]['occur_at']
        t1 = t0
        for e in i['action_sequence']:
            if e['event_type'] == 'ClosedEvent':
                t1 = e['occur_at']

        d_t = calculate_delta_t(t0, t1, unit='d')
        if d_t <= 1:
            continue
        res.append(i)
    write_json_list(res, data_dir + 'preprocessed_closed_issue_discussion_' + str(_time) + 'day.json')


def calcu_close_time():
    data_dir = get_global_val('data_dir')
    issues = load_json_list(data_dir + 'preprocessed_closed_issue_discussion.json')
    issue_close_duration = {}
    issue_close_duration_ratio = {}
    for i in issues:
        r = i['repo_name'].split('/')[1]
        n = i['target']['number']
        _id = r+'_'+str(n)

        issue_len = len(i['action_sequence'])
        t_0 = i['action_sequence'][0]['occur_at']
        t_1 = i['action_sequence'][issue_len-2]['occur_at']
        t_2 = i['action_sequence'][issue_len-1]['occur_at']
        if issue_len < 10:
            continue

        for k in range(issue_len-2, 0, -1):
            t_1 = i['action_sequence'][k]['occur_at']
            delta_t2 = calculate_delta_t(t_1, t_2, unit='s')
            if delta_t2 > 3:
                break

        delta_t1 = calculate_delta_t(t_0, t_1, unit='s')
        avg_t = delta_t1/(k)
        delta_t2 = calculate_delta_t(t_1, t_2, unit='s')
        issue_close_duration[_id] = delta_t2
        try:
            issue_close_duration_ratio[_id] = delta_t2/avg_t
        except Exception:
            print(_id, i['action_sequence'], issue_len, delta_t2, delta_t1)
        # for j in range(len(i['action_sequence'])-1, 0, -1):
        #     if i['action_sequence'][j]['event_type'] == 'ClosedEvent':
        #         break
        # if j == 1:
        #     k = 0
        # else:
        #     for k in range(j-1, 0, -1):
        #         if calculate_delta_t(i['action_sequence'][k]['occur_at'], i['action_sequence'][j]['occur_at'], unit='m') >= 1:
        #             break
        # try:
        #     d_t = calculate_delta_t(i['action_sequence'][k]['occur_at'], i['action_sequence'][j]['occur_at'], unit='h')
        # except Exception:
        #     print(j, k)
        #     print(i)
        #     exit(-1)
        # issue_close_duration[_id] = d_t

    write_json_dict(issue_close_duration, data_dir+'issue_last_close_duration.json')
    write_json_dict(issue_close_duration_ratio, data_dir + 'issue_last_close_duration_ratio.json')


def calcu_last_close_event(suffix):
    data_dir = get_global_val('data_dir')
    issues = load_json_list(data_dir + 'preprocessed_closed_issue_discussion_'+suffix+'.json')
    issue_close_duration = {}
    for i in issues:
        r = i['repo_name'].split('/')[1]
        n = i['target']['number']
        _id = r + '_' + str(n)
        for j in range(len(i['action_sequence']) - 1, 0, -1):
            if i['action_sequence'][j]['event_type'] == 'ClosedEvent':
                break
        e = i['action_sequence'][j-1]['event_type']
        if j == 1:
            k = 0
        else:
            for k in range(j - 1, 0, -1):
                if calculate_delta_t(i['action_sequence'][k]['occur_at'], i['action_sequence'][j]['occur_at'],
                                     unit='m') >= 1:
                    break
            e = i['action_sequence'][k]['event_type']
        issue_close_duration[_id] = e
    write_json_dict(issue_close_duration, data_dir + 'issue_last_close_event.json')


def calcu_fix_time():
    data_dir = get_global_val('data_dir')
    issues = load_json_list(data_dir + 'preprocessed_closed_issue_discussion.json')
    fix_time = {}
    for i in issues:
        t0 = i['action_sequence'][0]['occur_at']
        for e in range(len(i['action_sequence'])-1, 0, -1):
            # if i['action_sequence'][e]['actor'] not in bots:
            #     t1 = i['action_sequence'][e]['occur_at']
            #     break
            if i['action_sequence'][e]['event_type'] == 'ClosedEvent':
                t1 = i['action_sequence'][e]['occur_at']
                break
        d_t = calculate_delta_t(t0, t1, unit='d')
        _id = i['repo_name'].split('/')[1]+'_'+str(i['target']['number'])
        fix_time[_id] = d_t
    write_json_dict(fix_time, data_dir+'issue_fix_time.json')









def classify_sequence_by_avgtime(min_len=20):
    data_dir = get_global_val('data_dir')
    issues = load_json_list(data_dir + 'preprocessed_closed_issue_discussion.json')
    issue_time_level = {}
    for repo in ['ansible/ansible', 'tensorflow/tensorflow']:
        issue_avgtime = {}
        temp = []
        for i in issues:
            if i['repo_name'] not in repo:
                continue
            _id = i['repo_name'][0] + '_' + str(i['target']['number'])
            if len(i['action_sequence']) < min_len:
                continue
            t1 = i['action_sequence'][0]['occur_at']
            t2 = 0
            for e in i['action_sequence']:
                try:
                    if e['actor'] not in bots:
                        t2 = e['occur_at']
                except Exception:
                    pass

            delta_t = calculate_delta_t(t1, t2, unit='m')
            avg_time = delta_t/(len(i['action_sequence']))
            issue_avgtime[_id] = avg_time
            temp.append(avg_time)
        median = numpy.median(temp)
        for i in issue_avgtime:
            if issue_avgtime[i] < median:
                issue_time_level[i] = 'short'
            else:
                issue_time_level[i] = 'long'

        res = {'long': [], 'short': []}
        for i in issues:
            if i['repo_name'] not in repo:
                continue

            _id = i['repo_name'][0] + '_' + str(i['target']['number'])
            if _id not in issue_time_level:
                continue
            temp = {'_id': _id, 'action_sequence': []}
            for a in i['action_sequence']:
                temp['action_sequence'].append({'event_type': a['event_type'], 'occur_at': a['occur_at']})
            level = issue_time_level[_id]
            res[level].append(temp)
        repo = repo.split('/')[1]
        for level in ['long', 'short']:
            write_json_list(res[level], data_dir + 'sequences/issue_sequences_' + repo + '_' + level + '.json')


def classify_sequence_by_close_interval(suffix):
    data_dir = get_global_val('data_dir')
    issues = load_json_list(data_dir + 'preprocessed_closed_issue_discussion_'+suffix+'.json')
    close_interval = load_json_dict(data_dir+'issue_last_close_duration.json')
    for repo in ['ansible/ansible', 'tensorflow/tensorflow']:
        res = {'neg': [], 'pos': []}
        for i in issues:
            if i['repo_name'] not in repo:
                continue
            _id = i['repo_name'].split('/')[1] + '_' + str(i['target']['number'])
            level = None
            try:
                if close_interval[_id] <= 1:
                    level = 'pos'
                elif close_interval[_id] > 24:
                    level = 'neg'
            except Exception:
                continue
            if level is None:
                continue

            temp = {'_id': _id, 'action_sequence': []}
            for a in i['action_sequence']:
                temp['action_sequence'].append({'event_type': a['event_type'], 'occur_at': a['occur_at']})
            res[level].append(temp)

        repo = repo.split('/')[1]
        for level in ['neg', 'pos']:
            write_json_list(res[level], data_dir + 'sequences/issue_sequences_' + repo + '_' + level + '.json')


def classify_sequence_by_close_time():
    data_dir = get_global_val('data_dir')
    issues = load_json_list(data_dir + 'preprocessed_closed_issue_discussion.json')
    close_time = load_json_dict(data_dir + 'issue_last_close_duration.json')

    keys = set()
    for i in issues:
        _id = i['repo_name'].split('/')[1]+'_'+str(i['target']['number'])
        keys.add(_id)

    split_time = [3600*24, 3600*24]
    res = {'neg': [], 'pos': []}
    for i in issues:
        _id = i['repo_name'].split('/')[1] + '_' + str(i['target']['number'])
        level = None
        try:
            if close_time[_id] <= split_time[0]:
                level = 'pos'
            elif close_time[_id] > split_time[1]:
                level = 'neg'
        except Exception:
            continue
        if level is None:
            continue

        temp = {'_id': _id, 'action_sequence': []}
        for a in i['action_sequence']:
            temp['action_sequence'].append({'event_type': a['event_type'], 'occur_at': a['occur_at']})
        res[level].append(temp)

    for level in ['neg', 'pos']:
        write_json_list(res[level], data_dir + 'sequences/issue_sequences_tensorflow_' + level + '.json')


def classify_sequence():
    data_dir = get_global_val('data_dir')
    stalled_issues = load_json_list(data_dir + 'tensorflow_stalled_issues.json')
    failed_issues = load_json_list(data_dir + 'tensorflow_failed_issues.json')
    # resolved_issues = load_json_list(data_dir + 'tensorflow_resolved_issues_closebyself.json')
    resolved_issues = load_json_list(data_dir + 'tensorflow_resolved_issues.json')

    exist = set()
    res = {'neg': [], 'pos': []}
    for i in stalled_issues:
        _id = i['repo_name'].split('/')[1] + '_' + str(i['target']['number'])
        level = 'neg'
        temp = {'_id': _id, 'action_sequence': []}
        for j in range(len(i['action_sequence'])-4):
            temp['action_sequence'].append({'event_type': i['action_sequence'][j]['event_type'], 'occur_at': i['action_sequence'][j]['occur_at']})
        # for a in i['action_sequence']:
        #     # if a['actor'] not in bots:
        #     temp['action_sequence'].append({'event_type': a['event_type'], 'occur_at': a['occur_at']})
        res[level].append(temp)
        exist.add(_id)
    for i in resolved_issues:
        _id = i['repo_name'].split('/')[1] + '_' + str(i['target']['number'])
        level = 'pos'
        temp = {'_id': _id, 'action_sequence': []}
        for a in i['action_sequence']:
            # if a['actor'] not in bots:
            temp['action_sequence'].append({'event_type': a['event_type'], 'occur_at': a['occur_at']})
        res[level].append(temp)
        exist.add(_id)
    for i in failed_issues:
        _id = i['repo_name'].split('/')[1] + '_' + str(i['target']['number'])
        if _id in exist:
            continue
        level = 'neg'
        temp = {'_id': _id, 'action_sequence': []}
        for a in i['action_sequence']:
            # if a['actor'] not in bots:
            temp['action_sequence'].append({'event_type': a['event_type'], 'occur_at': a['occur_at']})
        res[level].append(temp)

    #
    # issues = load_json_list(data_dir + 'preprocessed_closed_issue_discussion.json')
    # close_time = load_json_dict(os.path.join(data_dir, 'issue_last_close_duration_ratio.json'))
    # close = []
    # for i in close_time:
    #     close.append(close_time[i])
    # qs = numpy.percentile(close, (25, 50, 75), method='midpoint')
    # iqr = qs[2] - qs[0]
    # _max = qs[2] + 3*iqr
    # for i in issues:
    #     _id = i['repo_name'].split('/')[1] + '_' + str(i['target']['number'])
    #     if _id in exist:
    #         continue
    #     if _id not in close_time:
    #         continue
    #     if close_time[_id] > _max:
    #         continue
    #     if close_time[_id] <= 0.25:
    #         level = 'pos'
    #     elif close_time[_id] >= 3:
    #         level = 'neg'
    #     else:
    #         continue
    #     temp = {'_id': _id, 'action_sequence': []}
    #     for a in i['action_sequence']:
    #         # if a['actor'] not in bots:
    #         temp['action_sequence'].append({'event_type': a['event_type'], 'occur_at': a['occur_at']})
    #     res[level].append(temp)

    for level in ['neg', 'pos']:
        write_json_list(res[level], data_dir + 'sequences/issue_sequences_tensorflow_' + level + '.json')


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


def generate_sequence():
    data_dir = get_global_val('data_dir')
    bug_fix = load_json_list(data_dir+'closed_bug_fix.json')
    efficiency = load_json_list(data_dir + 'bug_fix_with_efficiency.json')
    b_eff = {}
    for i in efficiency:
        try:
            _id = i['repo'][0] + '_' + str(i['number'])
            b_eff[_id] = i['efficiency']
        except Exception:
            pass  # median

    for repo in ['ansible/ansible', 'tensorflow/tensorflow']:
        sequences = {'high': [], 'low': []}
        for b in bug_fix:
            mention_time = set()
            if b['repo_name'] not in repo:
                continue
            temp = {'_id': b['repo_name'][0] + '_' + str(b['target']['number']), 'action_sequence': []}
            for a in b['action_sequence']:
                if a['event_type'] == 'MentionedEvent':
                    mention_time.add(a['occur_at'])
                temp['action_sequence'].append({'event_type': a['event_type'], 'occur_at': a['occur_at']})
            try:
                del_index = []
                index = 0
                for a in temp['action_sequence']:
                    if a['event_type'] == 'SubscribedEvent' and a['occur_at'] in mention_time:
                        del_index.append(index)
                    index += 1
                del_index.reverse()
                for i in del_index:
                    temp['action_sequence'].pop(i)
                eff = b_eff[temp['_id']]
                sequences[eff].append(temp)
            except Exception:
                pass  # median

        repo = repo.split('/')[1]
        for eff in sequences:
            write_json_list(sequences[eff], data_dir+'sequences/bug_fix_sequences_' + repo+'_'+eff + '.json')


def select_no_data_commits(commit_list):
    data_dir = get_global_val('data_dir')
    commits = load_json_dict(data_dir+'commits.json')
    exists = set()
    for c in commits:
        exists.add(c)
    res = commit_list-exists

    commit_additions = load_json_list(data_dir+'commit_diffs.json')
    for c in commit_additions:
        if c['sha'] in res:
            exists.add(c['sha'])

    # print(commit_list-exists)

    urls = []
    res = commit_list - exists
    res_c = {}
    with open(r'F:\data_back\BugEfficiency\data\issue_events.json', 'r') as f:
        for i in f:
            dic = json.loads(i)
            if dic['data']['__typename'] == 'ReferencedEvent':
                try:
                    res_c[dic['data']['commit']['oid']] = dic['data']['commitRepository']['nameWithOwner']
                except Exception:
                    pass
    with open(r'F:\data_back\BugEfficiency\data\pr_events.json', 'r') as f:
        for i in f:
            dic = json.loads(i)
            if dic['data']['__typename'] == 'ReferencedEvent':
                try:
                    res_c[dic['data']['commit']['oid']] = dic['data']['commitRepository']['nameWithOwner']
                except Exception:
                    pass

    for c in res:
        if c in res_c:
            _str = "https://api.github.com/repos/"+res_c[c]+"/commits/" + c + "?per_page=100"
            urls.append(_str)
        else:
            print(c)

    with open('commit_urls.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['_id'])
        for c in urls:
            writer.writerow([c])


def select_closed_issue():
    data_dir = get_global_val('data_dir')
    res = []
    with open(data_dir+'issue_discussion.json', 'r') as f:
        for i in f:
            dic = json.loads(i)
            if dic['state'] == 'Close' and dic['behavior_type'] == 'collective' and dic['type'] == 'Bug':
                res.append(dic)

    write_json_list(res, data_dir+'closed_bug_fix.json')


def calcu_inconsistent_ratio():
    repo = 'tensorflow'
    data_dir = get_global_val('result_dir')+'event_time_event/len20/'
    data = load_json_dict(data_dir+'entropy/input_sequences_'+repo+'.json')
    res_a = get_ratio(data, 20, 2)

    # data = load_json_dict(data_dir + 'event_interval/quartile/input_sequences_ansible.json')
    data = load_json_dict(data_dir + 'quartile/input_sequences_'+repo+'.json')
    res_b = get_ratio(data, 20, 2)
    # data = load_json_dict(data_dir + 'entropy_auto/input_sequences_ansible.json')
    # res_c = get_ratio(data, 25, 2)
    # df = pd.DataFrame({'x': range(2, 25), 'without time': res_a, 'quartile': res_b, 'IG': res_c})
    df = pd.DataFrame({'x': range(2, 20), 'IG': res_a, 'quartile': res_b})
    df = df.melt(id_vars=['x'], value_name='ratio', var_name='type')
    df = df.pivot(index='x', columns='type', values='ratio')
    print(df)
    # exit(-1)
    figure_dir = get_global_val('figure_dir')
    draw_line_plot(df, figure_dir+repo+'_inconsistent_ratio', 'inconsistent ratio in '+repo)


def get_ratio(data, N, split=1):
    res = []
    for n in range(2, N):
        # if n != 10:
        #     continue
        slow = []
        quick = []
        for d in data['neg']:
            d = ''.join(d)
            if len(d) < n*split:
                continue
            slow.append(d[0:n*split])
        for d in data['pos']:
            d = ''.join(d)
            if len(d) < n*split:
                continue
            quick.append(d[0:n*split])

        total_n = len(slow) * len(quick)
        _slow = set(slow)
        _quick = set(quick)

        intersect = _slow.intersection(_quick)

        count_1 = 0
        count_2 = 0
        for i in slow:
            if i in intersect:
                count_1 += 1
        for i in quick:
            if i in intersect:
                count_2 += 1

        inconsistent = count_1*count_2
        ratio = inconsistent/total_n
        # print("top {} : total sequences: {}, inconsistent ratio: {}".format(n, total_n, ratio))
        res.append(ratio)
        # if ratio < 0.01:
        #     break
    return res


def sequence_length_show():
    data_dir = get_global_val('data_dir') + 'sequences/'
    repo = 'tensorflow'
    data1 = load_json_list(data_dir+'issue_sequences_'+repo+'_pos.json')
    data2 = load_json_list(data_dir+'issue_sequences_'+repo+'_neg.json')
    lens = []
    for d in data1:
        lens.append(['pos', len(d['action_sequence'])])
    for d in data2:
        lens.append(['neg', len(d['action_sequence'])])

    df = pd.DataFrame(lens, columns=['type', 'length'])
    print(df)
    figure_dir = get_global_val('figure_dir')
    draw_histplot(df, figure_dir+repo+'_sequence_length', repo+' sequences length')

    draw_boxplot(df, figure_dir+repo+'_sequence_length_box', None)


def sequence_interval_show():
    data_dir = get_global_val('data_dir') + 'sequences/'
    data = load_json_dict(data_dir + 'input_sequences_ansible.json')


def generate_commit_loc():
    data_dir = get_global_val('data_dir')
    res = {}
    commits = load_json_list(data_dir+'commits.json')
    for i in commits:
        temp = { 'total': i['data']['stats']['total'],
                 'add': i['data']['stats']['additions'],
                 'del': i['data']['stats']['deletions'] }
        res[i['_id']] = temp


    commits = load_json_list(data_dir+'commit_diffs.json')
    for i in commits:
        if i['sha'] in res:
            continue
        temp = { 'total': i['data']['add']+i['data']['del'],
                 'add': i['data']['add'],
                 'del': i['data']['del'] }
        res[i['sha']] = temp

    write_json_dict(res, data_dir+'commits_loc.json')


def modify_pr_occur():
    data_dir = get_global_val('data_dir')
    data = load_json_list(data_dir+'c_bug_fix_with_loc.json')
    res = []
    for d in data:
        issue_num = d['target']['number']
        issue_repo = d['repo_name']
        for i in range(len(d['action_sequence'])):
            if d['action_sequence'][i]['event_type'] == 'PullRequestEvent':
                for e in d['action_sequence'][i]['sub_event']:
                    if e['event_type'] == 'CrossReferencedEvent' and e['supple_data']['number'] == issue_num and e['supple_data']['repo_name'] == issue_repo:
                        # print(d['action_sequence'][i]['occur_at'], e['occur_at'])
                        d['action_sequence'][i]['occur_at'] = e['occur_at']
                        break
        d['action_sequence'] = sorted(d['action_sequence'], key=lambda k: k['occur_at'])

        I_occur = 0
        for i in range(len(d['action_sequence'])):
            if d['action_sequence'][i]['event_type'] == 'RaiseIssueEvent':
                break
            I_occur += 1
        del d['action_sequence'][0:I_occur]

        for a in d['action_sequence']:
            if a['event_type'] in ['PullRequestEvent', 'ReferencedEvent']:
                res.append(d)
                break
    write_json_list(res, data_dir + 'c_bug_fix.json')


def translate_issue_body(data_path, write_path):
    res = []
    with open(data_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            _body = dic['data']['body']
            dic['data']['body'] = html_document_str(_body)
            res.append(dic)

    write_json_list(res, write_path)


# def html_to_str(r_data):
#     text = BeautifulSoup(r_data, "lxml").get_text()
#     return text


def html_document_str(doc):
    soup = BeautifulSoup(doc, features='lxml')
    # remove long code segment
    for i in soup.find_all('code'):
        i.replace_with(' ')
    for i in soup.find_all('pre'):
        i.replace_with(' ')
    # remove all quote
    for i in soup.find_all('blockquote'):
        i.replace_with(' ')
    doc = soup.get_text()
    return doc


def get_issue_body(selected_issues_num, data_path):
    res = {}
    with open(data_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            if dic['number'] not in selected_issues_num:
                continue
            _title = dic['data']['title']
            _body = dic['data']['body']
            _body = _body.replace("\n", " ")
            res[dic['number']] = {'title': _title, 'body': _body}
    return res


def add_comment_to_issues(issue_path, comment_path):
    issues = {}
    with open(issue_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            _id = dic['repo_name']+'_'+str(dic['number'])
            issues[_id] = dic

    with open(comment_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            _id = dic['repo_name'] + '_' + str(dic['number'])
            if _id not in issues:
                continue
            text = dic['data']['bodyHTML']
            created_at = dic['data']['createdAt']
            try:
                actor = dic['data']['author']['login']
            except Exception:
                actor = None
            temp = {'event_type': 'CommentEvent', 'text': text, 'created_at': created_at, 'actor': actor}
            issues[_id]['events'].append(temp)

    res = []
    for i in issues:
        issues[i]['events'] = sorted(issues[i]['events'], key=lambda k: k['created_at'])
        res.append(issues[i])
    write_json_list(res, issue_path)


def delete_merge_commit(commitDiff_path):
    diffs = load_json_list(commitDiff_path)
    res = []
    delete_commit = {'2f5d058d8f1e287d6c7e4257e64137fb0af7de0f',
                     'ea2e80888a68c399f422e6657913eb81973a9f9a',
                     '3b1a7ec090008169c840d91111690e9bc8ee6aa8',
                     'd437d4a559db77e6c8120d8240c7e6063abefcea',
                     '80e2ecc3c52021d8750c7376a3fc18c8e1b58e9f',
                     'e622028363baf48fa028b9ca86ebae0dc9dae772'}
    for d in diffs:
        if 'merge' in d['data']['msg'].lower() or 'merging' in d['data']['msg'].lower():
            continue
        elif 'rebas' in d['data']['msg'].lower():
            continue
        elif 'updat' in d['data']['msg'].lower() and 'from' in d['data']['msg'].lower():
            continue
        elif 'resolv' in d['data']['msg'].lower() and 'conflict' in d['data']['msg'].lower():
            continue
        elif d['sha'] in delete_commit:
            continue
        else:
            res.append(d)
    write_json_list(res, 'data/commit_diffs_limited.json')


def limit_commit_filetype(commitDiff_path):
    diffs = load_json_list(commitDiff_path)
    res = []

    d_suffix = {'md', 'rst', 'orig', 'lock', 'pub', 'stdout', 'stderr', 'csv', 'pbtxt', 'asciidoc', 'svg', 'templated'}
    c_suffix = {'json', 'txt'}
    # r_suffix = {'js', 'sha1', 'yml', 'url', 'sqlite3', 'exe', '1', 'py', 'yaml', 'network-jobarker', 'ts', 'html', 'tsx', '.service', 'h', 'java', 'sh',
    #             'env', 'j2', 'manifest', 'map', 'nix', 'cfg', 'localhost', 'example', 'inv', 'gni', 'mask', 'patch', 'ebuild', 'cs', 'scss', 'cpu', 'xml',
    #             'cc', 'ipynb', 'go', 'ps1', 'css', 'BUILD'}
    for c in diffs:
        if c['data']['del'] == 0 and c['data']['add'] > 500:
            continue
        _add = c['data']['add']
        _del = c['data']['del']
        for f in c['data']['files']:
            name = f['name'].split('/')
            name = name[len(name)-1].split('.')
            if len(name) >= 2:
                suf = name[len(name)-1]
                if suf in d_suffix:
                    _add = _add - f['add']
                    _del = _del - f['del']
                elif suf in c_suffix and f['add'] + f['del'] > 1000:
                    _add = _add - f['add']
                    _del = _del - f['del']
        c['data']['add'] = _add
        c['data']['del'] = _del
        res.append(c)

    write_json_list(res, 'data/commit_diffs_limited.json')


def classify_issues():
    data_dir = get_global_val("data_dir")
    figure_dir = get_global_val("figure_dir")
    data = load_json_dict(data_dir+'tensorflow_issues_len10_fix_time.json')
    fix_time = {}

    # res = []
    count = 0
    print(len(data))
    for i in data:
        _id = i
        year = data[i]['seq'][0]['occur_at'].split('-')[0]
        mon = data[i]['seq'][0]['occur_at'].split('-')[1]
        _time = data[i]['fix_time']
        if _time > 382.3086805555556*24*60:
            count += 1
            continue
        if year not in fix_time:
            fix_time[year] = {}
        if int(mon) not in fix_time[year]:
            fix_time[year][int(mon)] = {}
        fix_time[year][int(mon)][_id] = _time
        # res.append(_time/24/60)
    print(count)

    res = {}
    for year in fix_time:
        res[year] = []
        for mon in range(1, 13):
            if mon not in fix_time[year]:
                # res[year].append('')
                continue
            else:
                d = fix_time[year][mon]
                temp = []
                for i in d:
                    temp.append(d[i])
                _mean = numpy.mean(temp)
                # res[year].append(_mean)
                res[year] += temp
    frame = []
    for year in res:
        for d in res[year]:
            if d == '':
                continue
            frame.append([year, d])
    df = pd.DataFrame(frame, columns=['year', 'fix-time'])
    # df = pd.DataFrame(res)
    print(df)
    print(res)

    # f, ax = plt.subplots(figsize=(11, 6))
    # bp = sns.boxplot(x="year", y='fix-time', data=df, palette="Set3",  linewidth=1, showmeans=True)
    # # sns.lineplot(data=df, palette="Set2", linewidth=1)
    # # sns.despine(left=True, bottom=True)
    # plt.ylabel('fix-time')
    # plt.savefig(figure_dir+'tensorflow_issue_fix_time', dpi=150)

    nor_fix_time = []
    for year in res:
        _min = min(res[year])
        _max = max(res[year])
        for mon in fix_time[year]:
            for d in fix_time[year][mon]:
                temp = fix_time[year][mon][d]
                temp = (temp-_min)/(_max-_min)
                fix_time[year][mon][d] = temp
                nor_fix_time.append(temp)
    print(fix_time)
    qs = numpy.percentile(nor_fix_time, (25, 50, 75), method='midpoint')
    print(qs[0], qs[2])

    slow = []
    fast = []

    for year in fix_time:
        for mon in fix_time[year]:
            for d in fix_time[year][mon]:
                if fix_time[year][mon][d] <= qs[0]:
                    fast.append(d)
                elif fix_time[year][mon][d] >= qs[2]:
                    slow.append(d)

    res_slow = []
    res_fast = []
    for i in data:
        if i in slow:
            res_slow.append({'_id': i, 'data': data[i]})
        elif i in fast:
            res_fast.append({'_id': i, 'data': data[i]})

    write_json_list(res_slow, data_dir+'tensorflow_issues_slow.json')
    write_json_list(res_fast, data_dir + 'tensorflow_issues_fast.json')

    pos = []
    neg = []
    for i in res_slow:
        temp = {'_id': i['_id'], 'action_sequence': []}
        for a in i['data']['seq']:
            temp['action_sequence'].append({'event_type': a['event_type'], 'occur_at': a['occur_at']})
        neg.append(temp)
    for i in res_fast:
        temp = {'_id': i['_id'], 'action_sequence': []}
        for a in i['data']['seq']:
            temp['action_sequence'].append({'event_type': a['event_type'], 'occur_at': a['occur_at']})
        pos.append(temp)

    write_json_list(pos, data_dir + 'sequences/issue_sequences_tensorflow_pos.json')
    write_json_list(neg, data_dir + 'sequences/issue_sequences_tensorflow_neg.json')











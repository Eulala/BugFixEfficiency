from util import *
from bs4 import BeautifulSoup
import nltk
from output import *

# state_map = {'new': 1, 'comprehended': 2, 'assigned': 3, 'proposed': 4, 'passed': 5, 'closed': 6, 'failed': 7, 'discussed': 8}
bots = { 'tensorflowbutler', 'google-ml-butler', 'tensorflow-bot', 'ansible', 'ansibot'}


def extract_raw_data():
    mongo_config = get_global_val('mongo_config')
    data_dir = get_global_val('data_dir')
    mongo_c = MyMongo(mongo_config['ip'], mongo_config['username'], mongo_config['pwd'], port=int(mongo_config['port']))
    mongo_c.set_db_name(mongo_config['db_name'])
    mongo_c.connect()

    data = mongo_c.get_col_value(col_name='issue_discussion', cond={'repo_name': {"$in": ['pytorch/pytorch']}, 'behavior_type': 'collective'})
    write_json_list(data, data_dir + 'issue_discussion_test.json')
    mongo_c.close()

    # mongo_c = MyMongo(mongo_config['ip'], 'sbh', 'sbh123456', port=int(mongo_config['port']))
    # mongo_c.set_db_name('ghdb')
    # mongo_c.connect()
    # data = mongo_c.get_col_value(col_name='issueTimeline', cond={"index.repo_name": {"$in": ['pytorch', 'ansible']}})
    # write_json_list(data, data_dir+'issue_events_test.json')

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


def calculate_fix_time():
    data_dir = get_global_val('data_dir')
    data = load_json_list(data_dir+'closed_bug_fix.json')
    res = []
    for i in data:
        begin_at = i['occur_at']
        end_at = begin_at
        for e in i['action_sequence']:
            actor = e['actor']
            if actor is not None:
                if 'bot' in actor.lower() or actor in bots:
                    continue
            end_at = e['occur_at']

        delta_t = calculate_delta_t(begin_at, end_at)
        res.append({'repo': i['repo_name'], 'number': i['target']['number'], 'fix_time': delta_t})

    write_json_list(res, data_dir+'bug_fix_time.json')


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
    data_dir = get_global_val('data_dir')+'sequences/'
    data = load_json_dict(data_dir+'input_sequences_ansible.json')
    res_a = get_ratio(data, 25)

    data = load_json_dict(data_dir + 'event_interval/quartile/input_sequences_ansible.json')
    res_b = get_ratio(data, 25, 2)
    data = load_json_dict(data_dir + 'entropy_auto/input_sequences_ansible.json')
    res_c = get_ratio(data, 25, 2)
    df = pd.DataFrame({'x': range(2, 25), 'without time': res_a, 'quartile': res_b, 'IG': res_c})
    df = df.melt(id_vars=['x'], value_name='ratio', var_name='type')
    df = df.pivot(index='x', columns='type', values='ratio')

    figure_dir = get_global_val('figure_dir')
    draw_line_plot(df, figure_dir+'ansible_inconsistent_ratio')


def get_ratio(data, N, split=1):
    res = []
    for n in range(2, N):
        # if n != 10:
        #     continue
        slow = []
        quick = []
        for d in data['low']:
            d = ''.join(d)
            if len(d) < n*split:
                continue
            slow.append(d[0:n*split])
        for d in data['high']:
            d = ''.join(d)
            if len(d) < n*split:
                continue
            quick.append(d[0:n*split])

        slow = set(slow)
        quick = set(quick)
        total_n = len(slow)+len(quick)
        intersection = slow.intersection(quick)
        inconsistent = len(intersection)*2
        # intersection = set(slow).intersection(set(quick))
        # inconsistent = 0
        # for s in slow:
        #     if s in intersection:
        #         inconsistent += 1
        # for s in quick:
        #     if s in intersection:
        #         inconsistent += 1
        ratio = inconsistent/total_n
        # print("top {} : total sequences: {}, inconsistent ratio: {}".format(n, total_n, ratio))
        res.append(ratio)
        # if ratio < 0.01:
        #     break
    return res


def sequence_length_show():
    data_dir = get_global_val('data_dir') + 'sequences/'
    data = load_json_dict(data_dir + 'input_sequences_tensorflow.json')
    lens = []

    rename = {'high': 'fast', 'low': 'slow'}

    for d in data:
        for i in data[d]:
            lens.append([rename[d], len(i)])
    # print(lens)
    df = pd.DataFrame(lens, columns=['type', 'length'])
    # print(df)

    figure_dir = get_global_val('figure_dir')
    draw_histplot(df, figure_dir+'tensorflow_sequence_length', 'tensorflow sequences length')


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













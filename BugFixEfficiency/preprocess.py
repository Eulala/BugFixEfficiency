from util import *
from bs4 import BeautifulSoup
import nltk

# state_map = {'new': 1, 'comprehended': 2, 'assigned': 3, 'proposed': 4, 'passed': 5, 'closed': 6, 'failed': 7, 'discussed': 8}
mongo_config = {'ip': '', 'port': 0, 'username': '', 'pwd': '', 'db_name': ''}
data_dir = ''
commit_dir = ''


def initialize():
    config = load_config()
    global data_dir
    data_dir = config['DataPath']['root_dir']
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    global commit_dir
    commit_dir = config['DataPath']['commit_dir']
    if not os.path.exists(commit_dir):
        raise ValueError('no such commit dir')

    mongo = config['MongoDB']
    global mongo_config
    mongo_config['ip'] = mongo['ip']
    mongo_config['port'] = int(mongo['port'])
    mongo_config['username'] = mongo['username']
    mongo_config['pwd'] = mongo['pwd']
    mongo_config['db_name'] = mongo['db_name']


def extract_raw_data():
    # mongo_c = MyMongo(mongo_config['ip'], mongo_config['username'], mongo_config['pwd'], port=int(mongo_config['port']))
    # mongo_c.set_db_name(mongo_config['db_name'])
    # mongo_c.connect()
    #
    # data = mongo_c.get_col_value(col_name='bug_fix', cond={'repo_name': {"$in": ['tensorflow/tensorflow', 'ansible/ansible']}})
    # write_json_list(data, data_dir + 'bug_fix.json')
    # mongo_c.close()

    mongo_c = MyMongo(mongo_config['ip'], 'sbh', 'sbh123456', port=int(mongo_config['port']))
    mongo_c.set_db_name('ghdb')
    mongo_c.connect()
    # data = mongo_c.get_col_value(col_name='issueTimeline', cond={"index.repo_name": {"$in": ['tensorflow', 'ansible']}})
    # write_json_list(data, data_dir+'issue_events.json')

    data = mongo_c.get_col_value(col_name='pullRequestTimeline', cond={"index.repo_name": {"$in": ['tensorflow', 'ansible']}})
    write_json_list(data, data_dir+'pr_events.json')
    mongo_c.close()


def find_commit_repo():
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
    res = {}
    files = os.listdir(commit_dir)
    files = list(filter(lambda f: 'enrich' not in f, files))
    for f in files:
        commits = load_from_disk(commit_dir+f)
        for c in tqdm.tqdm(commits):
            if c in commit_list:
                res[c] = commits[c]

    write_json_dict(res, data_dir+'commits.json')


def select_no_data_commits(commit_list):
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
    res = []
    with open(data_dir+'bug_fix.json', 'r') as f:
        for i in f:
            dic = json.loads(i)
            if dic['state'] == 'Close':
                res.append(dic)

    write_json_list(res, data_dir+'closed_bug_fix.json')


def generate_commit_loc():
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



# def extract_raw_data(project_name):
#     issues = extract_data('issue', {'proj_name': project_name, 'collection': 'issue'})
#     write_json_data(issues, 'data/issues.json')
#     prs = extract_data('pr', {'proj_name': project_name, 'collection': 'pullRequest'})
#     write_json_data(prs, 'data/prs.json')
#     issue_label = extract_data('label', {'proj_name': project_name, 'collection': 'issueLabel'})
#     write_json_data(issue_label, 'data/issue_labels.json')
#     issue_event = extract_data('event', {'proj_name': project_name, 'collection': 'issueTimeline'})
#     write_json_data(issue_event, 'data/issue_events.json')
#     pr_event = extract_data('event', {'proj_name': project_name, 'collection': 'pullRequestTimeline'})
#     write_json_data(pr_event, 'data/pr_events.json')
#     issue_comment = extract_data('comment', {'proj_name': project_name, 'collection': 'issueComment'})
#     write_json_data(issue_comment, 'data/issue_comments.json')
#     pr_comment = extract_data('comment', {'proj_name': project_name, 'collection': 'pullRequestComment'})
#     write_json_data(pr_comment, 'data/pr_comments.json')
#     issue_body = extract_data('body', {'proj_name': project_name, 'collection': 'issue'})
#     write_json_data(issue_body, 'data/issue_bodies.json')



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


def add_commitDiff_to_issues():
    commit_loc = load_json_dict(data_dir+'commits_loc.json')
    bug_fix = load_json_list(data_dir+'closed_bug_fix.json')

    res = []
    for i in bug_fix:
        i['LOC'] = 0
        for e in range(0, len(i['action_sequence'])):
            if i['action_sequence'][e]['event_type'] == 'ReferencedEvent':
                i['action_sequence'][e]['supple_data']['loc'] = 0
                try:
                    sha = i['action_sequence'][e]['supple_data']['oid']
                    i['action_sequence'][e]['supple_data']['loc'] = commit_loc[sha]['total']
                except:
                    pass
                i['LOC'] += i['action_sequence'][e]['supple_data']['loc']
            elif i['action_sequence'][e]['event_type'] == 'PullRequestEvent':
                sub_events = i['action_sequence'][e]['sub_event']
                for m in range(0, len(sub_events)):
                    if sub_events[m]['event_type'] == 'PullRequestCommit':
                        sub_events[m]['supple_data']['loc'] = 0
                        try:
                            sha = sub_events[m]['supple_data']['oid']
                            sub_events[m]['supple_data']['loc'] = commit_loc[sha]['total']
                        except:
                            pass
                        i['LOC'] += sub_events[m]['supple_data']['loc']
                i['action_sequence'][e]['sub_event'] = sub_events

        res.append(i)
    write_json_list(res, data_dir+'c_bug_fix_with_loc.json')






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








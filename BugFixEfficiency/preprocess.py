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
    mongo_c = MyMongo(mongo_config['ip'], mongo_config['username'], mongo_config['pwd'], port=int(mongo_config['port']))
    mongo_c.set_db_name(mongo_config['db_name'])
    mongo_c.connect()

    data = mongo_c.get_col_value(col_name='bug_fix', cond={'repo_name': {"$in": ['tensorflow/tensorflow', 'ansible/ansible']}})
    write_json_list(data, data_dir + 'bug_fix.json')


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


def select_bug_issue():
    res = []
    issues = {}
    with open('data/issues.json', 'r') as f:
        for i in f:
            dic = json.loads(i)
            _id = dic['repo_name']+'_'+str(dic['number'])
            issues[_id] = dic

    bug_set = set()
    with open('data/issue_labels.json', 'r') as f:
        for i in f:
            dic = json.loads(i)
            if "bug" in dic['name']:
                bug_set.add(dic['name'])
                _id = dic['repo_name']+'_'+str(dic['number'])
                try:
                    res.append(issues[_id])
                except Exception:
                    print(_id)

    # print(bug_set)
    write_json_list(res, 'data/bug_issues.json')


def add_event_to_issues(issue_path, event_path, _type):
    issues = {}
    with open(issue_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            dic['events'] = []
            _id = dic['repo_name'] + '_' + str(dic['number'])
            issues[_id] = dic


    with open(event_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            _id = dic['repo_name'] + '_' + str(dic['number'])
            if _id not in issues:
                continue
            try:
                temp = {'event_type': dic['data']["__typename"], 'created_at': dic['data']['createdAt']}
                try:
                    temp['actor'] = dic['data']['actor']['login']
                except Exception:
                    temp['actor'] = None
            except Exception:
                if dic['data']["__typename"] == 'PullRequestCommit':
                    temp = {'event_type': dic['data']["__typename"], 'created_at': '1970-01-01', 'commit': dic['data']['commit']}
                    issues[_id]['events'].append(temp)
                    continue


            if dic['data']["__typename"] in ["AssignedEvent", 'UnassignedEvent']:
                if _type == 'issue':
                    temp['event_type'] = 'I_' + dic['data']["__typename"]
                else:
                    temp['event_type'] = 'P_' + dic['data']["__typename"]
                try:
                    temp['assignee'] = dic['data']['assignee']['login']
                except Exception:
                    temp['assignee'] = None
            elif dic['data']["__typename"] == "CrossReferencedEvent":
                if dic['data']['isCrossRepository'] is True:
                    continue
                if _type == 'issue':
                    temp['linked_pr'] = dic['data']['source']['number']
                else:
                    temp['linked_issue'] = dic['data']['source']['number']
            elif dic['data']["__typename"] in ['ReferencedEvent']:
                try:
                    temp['commit'] = dic['data']['commit']
                except Exception:
                    continue
            elif dic['data']["__typename"] in ['LabeledEvent', 'UnlabeledEvent']:
                temp['label'] = dic['data']['label']['name']
            elif dic['data']['__typename'] in ['ClosedEvent', 'MergedEvent', 'ReopenedEvent', 'MarkedAsDuplicateEvent', 'HeadRefForcePushedEvent', 'BaseRefForcePushedEvent']:
                if _type == 'issue':
                    temp['event_type'] = 'I_' + dic['data']["__typename"]
                else:
                    temp['event_type'] = 'P_' + dic['data']["__typename"]
            elif dic['data']['__typename'] in ['IssueComment']:
                continue
            # else:
                # continue
            issues[_id]['events'].append(temp)

    res = []
    for i in issues:
        issues[i]['events'] = sorted(issues[i]['events'], key=lambda k: k['created_at'])
        res.append(issues[i])
    if _type == 'issue':
        write_json_list(res, 'data/bug_issues_with_events.json')
    else:
        write_json_list(res, 'data/prs_with_events.json')


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


def select_issue_with_code(issue_path, pr_path, write_path):
    pr_set = set()
    with open(pr_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            _id = dic['repo_name'] + '_' + str(dic['number'])
            pr_set.add(_id)
    issue_set = set()
    with open(issue_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            _id = dic['repo_name'] + '_' + str(dic['number'])
            issue_set.add(_id)
    issues_linked_pr = get_cross_reference(issue_path, pr_set)
    prs_linked_issue = get_cross_reference(pr_path, issue_set)
    for i in prs_linked_issue:
        repo = i.split('_')[0]
        number = int(i.split('_')[1])
        for u in prs_linked_issue[i]:
            _id = repo+'_'+str(u)
            if _id not in issues_linked_pr:
                issues_linked_pr[_id] = set()
            issues_linked_pr[_id].add(number)
    issues_commit = get_issue_commit(issue_path)
    prs_commit = get_issue_commit(pr_path, _type='pr')
    res = []
    with open(issue_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            _id = dic['repo_name'] + '_' + str(dic['number'])
            dic['linked_pr'] = []
            dic['linked_commit'] = set()
            if _id in issues_commit and len(issues_commit[_id]) > 0:
                dic['linked_commit'] = issues_commit[_id]
            if _id in issues_linked_pr and len(issues_linked_pr[_id]) > 0:
                dic['linked_pr'] = list(issues_linked_pr[_id])
                for p in dic['linked_pr']:
                    _id = dic['repo_name'] + '_' + str(p)
                    try:
                        dic['linked_commit'] = dic['linked_commit'].union(prs_commit[_id])
                    except Exception:
                        continue
            dic['linked_commit'] = list(dic['linked_commit'])
            if len(dic['linked_commit']) > 0:
                res.append(dic)
    write_json_list(res, write_path)



def get_cross_reference(issue_path, pr_set):
    issues_link = {}
    with open(issue_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            _id = dic['repo_name'] + '_' + str(dic['number'])
            for e in dic['events']:
                if e['event_type'] == 'CrossReferencedEvent':
                    if _id not in issues_link:
                        issues_link[_id] = set()
                    if 'linked_pr' in e:
                        pr_id = dic['repo_name'] + '_'+str(e['linked_pr'])
                        if pr_id in pr_set:
                            issues_link[_id].add(e['linked_pr'])
                    elif 'linked_issue' in e:
                        pr_id = dic['repo_name'] + '_' + str(e['linked_issue'])
                        if pr_id in pr_set:
                            issues_link[_id].add(e['linked_issue'])
    return issues_link


def get_issue_commit(issue_path, _type='issue'):
    issues_commit = {}
    data = load_json_list(issue_path)
    for dic in data:
        _id = dic['repo_name'] + '_'+str(dic['number'])
        if _id not in issues_commit:
            issues_commit[_id] = set()
        for e in dic['events']:
            if _type == 'issue':
                if e['event_type'] == 'ReferencedEvent' and 'commit' in e:
                    issues_commit[_id].add(e['commit']['oid'])
            elif _type == 'pr':
                if e['event_type'] == 'PullRequestCommit' and 'commit' in e:
                    issues_commit[_id].add(e['commit']['oid'])
    return issues_commit



def map_cross_reference(issues_link, prs_link):
    # print(len(issues_link))
    issue_pr_map = {}
    modified_issues_link = {}
    for i in issues_link:
        modified_issues_link[i] = []
        for j in issues_link[i]:
            if j > i:
                modified_issues_link[i].append(j)
        if len(modified_issues_link[i]) == 1:
            issue_pr_map[i] = modified_issues_link[i]
        elif len(modified_issues_link[i]) > 1:
            # print(i, modified_issues_link[i])
            for j in modified_issues_link[i]:
                if j in prs_link and i in prs_link[j]:
                    if i in issue_pr_map:
                        issue_pr_map[i].append(j)
                    else:
                        issue_pr_map[i] = [j]
    return issue_pr_map


def select_closed_issue(path):
    res = []
    with open(path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            flag = False
            for e in dic['events']:
                if 'ClosedEvent' in e['event_type']:
                    flag = True
                elif 'ReopenedEvent' in e['event_type']:
                    flag = False
            if flag:
                res.append(dic)
    write_json_list(res, 'data/closed_bug_issues.json')


def add_commitDiff_to_issues(issue_path, commitDiff_path, write_path):
    issues = load_json_list(issue_path)
    abnormal_issues = load_json_list('data/abnormal_issues.json')
    delete_issues = {}
    for i in abnormal_issues:
        delete_issues[i['number']] = i['repo_name']
    commitDiffs = {}
    with open(commitDiff_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            commitDiffs[dic['sha']] = {'add': dic['data']['add'], 'del': dic['data']['del']}
    res = []
    not_found = set()
    for i in issues:
        if i['number'] in delete_issues and i['repo_name'] == delete_issues[i['number']]:
            continue
        i['LOC'] = []
        for c in i['linked_commit']:
            try:
                i['LOC'].append(commitDiffs[c])
            except Exception:
                i['LOC'].append({'add': 0, 'del': 0})
                not_found.add(c)
        res.append(i)
    print(len(not_found))
    # write_json_data(list(not_found), 'data/can_not_found_commit.json')
    write_json_list(res, write_path)


def integrate_issue_and_prs(issue_path, pr_path, write_path):
    prs = {}
    with open(pr_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            _id = dic['repo_name']+'_'+str(dic['number'])
            prs[_id] = dic
    res_i = []
    issues = load_json_list(issue_path)
    for i in issues:
        for p in i['linked_pr']:
            _id = i['repo_name']+'_'+str(p)
            temp = { 'event_type': 'PullRequestEvent', 'created_at': prs[_id]['created_at'], 'actor': prs[_id]['author']}
            i['events'].append(temp)
        res_e = []
        for e in i['events']:
            if e['event_type'] == 'CrossReferencedEvent' and e['linked_pr'] in i['linked_pr']:
                continue
            res_e.append(e)
        i['events'] = sorted(res_e, key=lambda k: k['created_at'])
        res_i.append(i)

    write_json_list(res_i, write_path)


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








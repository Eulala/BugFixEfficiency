from util import *

mongo_ip = '172.27.135.32'
mongo_name = 'sbh'
mongo_pwd = 'sbh123456'
db_name = 'ghdb'


def generate_commit_diffs():
    client = pymongo.MongoClient(mongo_ip, 27017)
    client.sponsor.authenticate('sponsor', 'sponsor', mechanism='SCRAM-SHA-1')
    db = client['sponsor']
    col_name = 'CommitDetails'
    res = {}
    for r in db[col_name].find():
        c = r['contributions']
        if 'sha' in c:
            sha = c['sha']
            res[sha] = {'add': c['stats']['additions'], 'del': c['stats']['deletions'], 'msg': c['commit']['message']}
            files = c['files']
            res[sha]['files'] = []
            for f in files:
                res[sha]['files'].append({'name': f['filename'], 'add': f['additions'], 'del': f['deletions']})
    client.close()

    data = []
    for r in res:
        data.append({'sha': r, 'data': res[r]})
    write_json_data(data, 'data/commit_diffs.json')





def extract_data(data_type, args):
    client = pymongo.MongoClient(mongo_ip, 27017)
    client.ghdb.authenticate(mongo_name, mongo_pwd, mechanism='SCRAM-SHA-1')
    db = client[db_name]
    res = func_dict.get(data_type, func_none)(db, args)
    client.close()
    return res


def extract_issue(db, args):
    res = []
    col_name = args['collection']
    collection = db[col_name]
    proj_name = args['proj_name']
    for r in collection.find({"index.repo_name": {"$in": proj_name}}):
        issue = { 'number': r['index']['number'], 'repo_name': r['index']['repo_name'], 'created_at': r['data']['repository'][col_name]['createdAt']}
        try:
            issue['author'] = r['data']['repository'][col_name]['author']['login']
        except Exception:
            issue['author'] = None
        res.append(issue)
    return res


def extract_label(db, args):
    res = []
    col_name = args['collection']
    collection = db[col_name]
    proj_name = args['proj_name']
    for r in collection.find({"index.repo_name": {"$in": proj_name}}):
        label = { 'number': r['index']['number'], 'repo_name': r['index']['repo_name'], 'name': r['data']['name']}
        res.append(label)
    return res


def extract_event(db, args):
    res = []
    col_name = args['collection']
    collection = db[col_name]
    proj_name = args['proj_name']
    for r in collection.find({"index.repo_name": {"$in": proj_name}}):
        # if r['data']['__typename'] in args['event']:
        event = {'number': r['index']['number'], 'repo_name': r['index']['repo_name'], 'data': r['data']}
        res.append(event)
    return res


def extract_comment(db, args):
    res = []
    col_name = args['collection']
    collection = db[col_name]
    proj_name = args['proj_name']
    for r in collection.find({"index.repo_name": {"$in": proj_name}}):
        # if r['data']['__typename'] in args['event']:
        comment = { 'number': r['index']['number'], 'repo_name': r['index']['repo_name'], 'data': r['data'] }
        res.append(comment)
    return res


def extract_body(db, args):
    res = []
    col_name = args['collection']
    collection = db[col_name]
    proj_name = args['proj_name']
    for r in collection.find({"index.repo_name": {"$in": proj_name}}):
        i_body = r['data']['repository']['issue']['bodyHTML']
        i_title = r['data']['repository']['issue']['titleHTML']
        body = {'number': r['index']['number'], 'repo_name': r['index']['repo_name'], 'data': {'title': i_title, 'body': i_body}}
        res.append(body)
    return res


def extract_commitDiff(db, args):
    res = []
    col_name = args['collection']
    collection = db[col_name]
    oids = args['oid_set']
    for r in collection.find({"oid": {"$in": oids}}):
        additions = r['add']
        deletions = r['del']
        res.append({'oid': r['oid'], 'loc': {'add': additions, 'del': deletions}})
    return res


func_dict = {'issue': extract_issue, 'pr': extract_issue, 'label': extract_label, 'event': extract_event, 'comment': extract_comment, 'body': extract_body, 'commitDiff': extract_commitDiff }



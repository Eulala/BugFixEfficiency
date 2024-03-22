import json
import time
import pymongo


class MyMongo(object):
    client = None
    port = 0
    db_name = ''

    def __init__(self, ip, name, pwd, port=27017):
        self.ip = ip
        self.name = name
        self.pwd = pwd
        self.port = port

    def connect(self):
        try:
            client = pymongo.MongoClient(self.ip, self.port, serverSelectionTimeoutMS=5000)
            client.get_database(self.db_name).authenticate(self.name, self.pwd, mechanism='SCRAM-SHA-1')
            self.client = client
            return True
        except Exception:
            return False

    def close(self):
        self.client.close()

    def set_db_name(self, db_name):
        self.db_name = db_name

    def get_db(self):
        return self.client[self.db_name]

    def get_col_value(self, col_name, cond):
        db = self.get_db()
        col = db[col_name]
        res = []
        for c in col.find(cond):
            c.pop('_id')
            res.append(c)
        return res


def write_json_list(data, filename):
    start = time.time()
    with open(filename, 'w') as f:
        for i in data:
            f.write(json.dumps(i)+'\n')
    end = time.time()
    print('write {} lines to {} runtime: {}'.format(len(data), filename, end-start))


def load_json_list(filename):
    start = time.time()
    data = []
    with open(filename, 'r') as f:
        for i in f:
            dic = json.loads(i)
            data.append(dic)
    end = time.time()
    print('load {} lines from {} runtime: {}'.format(len(data), filename, end-start))
    return data


def export_repo_data(_type, repo_list, write_path):
    mongo_c = MyMongo('172.27.135.32', 'sbh', 'sbh123456', port=27017)
    mongo_c.set_db_name('ghdb')
    mongo_c.connect()

    data = []
    for repo in repo_list:
        data += mongo_c.get_col_value(col_name=_type,
                                      cond={'index.repo_name': repo[0], 'index.repo_owner': repo[1]})
    write_json_list(data, write_path)
    mongo_c.close()


def export_user_info(users, write_path):
    mongo_c = MyMongo('172.27.135.32', 'sbh', 'sbh123456', port=27017)
    mongo_c.set_db_name('ghdb')
    mongo_c.connect()

    data = mongo_c.get_col_value(col_name='login',
                                 cond={'index.login': {"$in": users}})
    write_json_list(data, write_path)
    mongo_c.close()


def export_data():
    repo_list = [['tensorflow', 'tensorflow'], ['ansible', 'ansible']]
    export_repo_data('issue', repo_list, 'issue.json')
    export_repo_data('pullRequest', repo_list, 'pr.json')
    export_repo_data('issueTimeline', repo_list, 'issue_event.json')
    export_repo_data('pullRequestTimeline', repo_list, 'pr_event.json')
    export_repo_data('commit', repo_list, 'commit.json')

    user_list = list(generate_user_login())
    export_user_info(user_list, 'users.json')
    user_name = generate_user_name('users.json')
    write_json_list(user_name, 'user_name.json')


def generate_user_login():
    user_login = set()
    data = load_json_list('issue.json')
    for d in data:
        login = d['data']['repository']['issue']['author']['login']
        user_login.add(login)
    return user_login


def generate_user_name(filename):
    user_name = {}
    data = load_json_list(filename)
    for d in data:
        if d['data']['repositoryOwner']['__typename'] == 'User':
            login = d['data']['repositoryOwner']['login']
            name = d['data']['repositoryOwner']['name']
            loc = d['data']['repositoryOwner']['location']
            user_name[login] = {'name': name, 'location': loc}
    return user_name

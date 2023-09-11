from util import *
import pandarallel
from sgt import SGT
from sklearn.decomposition import PCA
import pm4py


def transform_loading_data():
    data_dir = get_global_val('data_dir') + 'sequences/entropy_auto/'
    data = load_json_data(data_dir+'ansible_neg_0.1_sup_csp.json')
    res = []
    sequences = []
    count = 0
    for d in data:
        d['number'] = count
        res.append(d)
        # seq = "".join(d['seq'])
        sequences.append([count, d['seq']])
        count += 1
    write_json_data(res, data_dir+'ansible_neg_0.1_sup_csp_trans.json')
    #
    # with open(r'data/closed_bug_fix_sequences.json', 'r') as f:
    #     for i in f:
    #         dic = json.loads(i)
    #         for s in dic['sequences']:
    #             for m in dic['sequences'][s]:
    #                 if m['number'] in exist_:
    #                     continue
    #                 temp_list = [m['number'], list(m['sequence'])]
    #                 sequences.append(temp_list)
    #                 exist_.add(m['number'])
    corpus = pd.DataFrame(sequences, columns=['id', 'sequence'])
    # corpus.to_csv(r'data/closed_bug_fix_sequences.csv')
    return corpus


def compute_sgt_embeddings(sequences):
    sgt_ = SGT(kappa=1,
               lengthsensitive=True,
               mode='multiprocessing')
    sgt_embedding_df = sgt_.fit_transform(sequences)

    sgt_embedding_df = sgt_embedding_df.set_index('id')
    # print(sgt_embedding_df)
    return sgt_embedding_df


def perform_pca(sgt_embedding_df):
    # for comp in range(50, 150):
    #     pca = PCA(n_components=comp)
    #     pca.fit(sgt_embedding_df)
    #     print("n_components = {}, ratio = {}".format(comp, numpy.sum(pca.explained_variance_ratio_)))
    comp = 50
    pca = PCA(n_components=comp)
    pca.fit(sgt_embedding_df)
    print("n_components = {}, ratio = {}".format(comp, numpy.sum(pca.explained_variance_ratio_)))
    X = pca.transform(sgt_embedding_df)
    # print(X)
    df = pd.DataFrame(data=X)
    # print(df)
    return df


def clustering(df):
    kmeans = KMeans(n_clusters=5)  # n_clusters:number of cluster
    kmeans.fit(df)
    # print(kmeans.labels_)
    label = {}
    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i] not in label:
            label[kmeans.labels_[i]] = []
        label[kmeans.labels_[i]].append(i)

    data_dir = get_global_val('data_dir') + 'sequences/entropy_auto/'
    data = load_json_data(data_dir + 'ansible_neg_0.1_sup_csp_trans.json')
    seqs = {}
    for d in data:
        seqs[d['number']] = d['seq']
    seq_cluster = []
    for i in range(len(label)):
        seq_cluster.append([])
        for j in label[i]:
            seq_cluster[i].append(seqs[j])

    print(seq_cluster)
    write_json_list(seq_cluster, data_dir+'ansible_neg_0.1_sup_csp_clusters.json')
    # print(kmeans.cluster_centers_)




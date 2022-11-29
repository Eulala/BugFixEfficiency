from util import *
import pandarallel
from sgt import SGT
from sklearn.decomposition import PCA


def transform_loading_data():
    exist_ = set()
    sequences = []
    with open(r'data/closed_bug_fix_sequences.json', 'r') as f:
        for i in f:
            dic = json.loads(i)
            for s in dic['sequences']:
                for m in dic['sequences'][s]:
                    if m['number'] in exist_:
                        continue
                    temp_list = [m['number'], list(m['sequence'])]
                    sequences.append(temp_list)
                    exist_.add(m['number'])
    corpus = pd.DataFrame(sequences, columns=['id', 'sequence'])
    # corpus.to_csv(r'data/closed_bug_fix_sequences.csv')
    return corpus


def compute_sgt_embeddings(sequences):
    sgt_ = SGT(kappa=1,
               lengthsensitive=True,
               mode='multiprocessing')
    sgt_embedding_df = sgt_.fit_transform(sequences)

    sgt_embedding_df = sgt_embedding_df.set_index('id')
    print(sgt_embedding_df)
    return sgt_embedding_df


def perform_pca(sgt_embedding_df):
    for comp in range(50, 150):
        pca = PCA(n_components=comp)
        pca.fit(sgt_embedding_df)
        print("n_components = {}, ratio = {}".format(comp, numpy.sum(pca.explained_variance_ratio_)))

    # X = pca.transform(sgt_embedding_df)
    # print(X)
    # df = pd.DataFrame(data=X, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'])
    # print(df)
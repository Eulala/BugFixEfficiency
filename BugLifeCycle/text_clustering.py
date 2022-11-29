import nltk
import re
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from typing import Union, Dict, Tuple, List, Callable


from util import *
from gensim.models import Word2Vec, word2vec
import numpy as np
from sklearn.cluster import KMeans



def nltk_download():
    # nltk.download("stopwords")
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')


def generate_tokens(text, tokenizer, wnl):
    # nltk_download()
    res = []
    sentences = nltk.sent_tokenize(text)
    for s in sentences:
        tokens = tokenizer.tokenize(s)
        pos_tags = nltk.pos_tag(tokens)
        lemmas_sent = []
        for tag in pos_tags:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        res = res + lemmas_sent
    return res


def text_to_vector(issue_contents):
    # # tokenize:
    # tokenizer = nltk.RegexpTokenizer(r'\w+')
    # wnl = WordNetLemmatizer()
    # stop_words = stopwords.words('english')
    #
    # issue_contents_tokens = {}
    # for i in issue_contents:
    #     title = issue_contents[i]['title']
    #     body = issue_contents[i]['body']
    #     text = title + ' ' + body
    #     text_tokens = generate_tokens(text, tokenizer, wnl)
    #     filtered_words = [word for word in text_tokens if word not in stop_words]
    #     lowercase_words = [word.lower() for word in filtered_words]
    #     issue_contents_tokens[i] = lowercase_words
    #
    # with open('data/tensorflow_issue_body_tokens.json', 'w') as f:
    #     for i in issue_contents_tokens:
    #         f.write(json.dumps({'number': i, 'content_tokens': issue_contents_tokens[i]})+'\n')
    #
    # with open('data/issue_contents_tokens.txt', 'w', newline='', encoding='utf-8') as f:
    #     for i in issue_contents_tokens:
    #         line = " ".join(issue_contents_tokens[i])
    #         f.write(line + '\n')
    #
    # # word2vec model training:
    # sentences = word2vec.LineSentence('data/issue_contents_tokens.txt')
    # w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    # w2v_model.save('word2vec.model')


    # load model, calculate the average word vector
    model = Word2Vec.load('word2vec.model')
    input_map = {}
    corpus_input = []
    count = 0
    # for i in issue_contents_tokens:
    #     corpus_input.append(issue_contents_tokens[i])
    #     input_map[i] = count
    #     count = count + 1

    with open('data/tensorflow_issue_body_tokens.json', 'r') as f:
        for i in f:
            dic = json.loads(i)
            corpus_input.append(dic['content_tokens'])
            input_map[dic['number']] = count
            count = count + 1
    res = averaged_word_vectorizer(corpus_input, model=model, num_features=100)
    return res, input_map


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype='float64')
    n_words = 0
    for word in words:
        if word in vocabulary:
            n_words = n_words + 1
            feature_vector = np.add(feature_vector, model.wv[word])
    if n_words:
        feature_vector = np.divide(feature_vector, n_words)
    return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
    # get the all vocabulary
    vocabulary = set(model.wv.index_to_key)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in
                corpus]
    return np.array(features)


def text_clustering(text_vec, text_map, write_path):
    # num_map = {}
    # num_count = 0
    # for i in range(0, len(text_vec)):
    #     num_map[text_map[i]] = num_count
    #     data_x.append(text_vec[i])
    #     num_count = num_count+1
    # print(data_x)
    data_x = numpy.array(text_vec)
    print(data_x.shape)

    kmeans = KMeans(n_clusters=4)  # n_clusters:number of cluster
    kmeans.fit(data_x)
    # print(kmeans.labels_)
    count = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in kmeans.labels_:
        count[i] = count[i]+1
    print(count)
    for i in text_map:
        text_map[i] = str(kmeans.labels_[text_map[i]])

    with open(write_path, 'w') as f:
        f.write(json.dumps(text_map))


import os.path

import numpy

from preprocess import *
from process_mining import *
from bug_fix_efficiency_classify import *
from contrast_sequential_pattern_mining import *
import matplotlib.pyplot as plt
from sequence_cluster import *
from text_clustering import *
from pre_classification import *
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def conduct_CDSPM_class3(repo_name, max_t, min_len, max_len):
    model_sequence(repo_name)

    data_dir = get_global_val('result_dir')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    cut_sequence(min_len, max_len, max_t, repo_name, interval=3)

    # cut_sequence_by_time(30, repo_name)

    # cut_sequence_by_time(_time=1, interval=3)
    # time_discretize(data_dir,  pair=True)

    X, Y, D = generate_dataset(repo_name)
    data_dir = os.path.join(get_global_val('result_dir'), "{}_{}_{}_new".format(repo_name, str(min_len), str(max_len)))
    # data_dir = os.path.join(get_global_val('result_dir'), "{}_{}_{}_paras".format(repo_name, str(min_len), str(max_len)))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # data_dir = os.path.join(data_dir, "sup{}_gr{}".format(0.1, 1.75))
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    write_json_list(D, os.path.join(data_dir, 'all_sequences.json'))

    # d = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=10)
    d = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)

    total_test_2 = {'fast': [], 'slow': []}
    total_pred_2 = {'fast': [], 'slow': []}

    total_test_3 = []
    total_pred_3 = []
    count = 1
    for train_idx, test_idx in d.split(X, Y):
        x_train, x_test = numpy.array(X, dtype=object)[train_idx], numpy.array(X, dtype=object)[test_idx]
        y_train, y_test = numpy.array(Y, dtype=object)[train_idx], numpy.array(Y, dtype=object)[test_idx]
        # dataset_time_discretize_quartile(x_train, y_train, data_dir)
        dataset_time_discretize(x_train, y_train, data_dir)
        # dataset_time_discretize_back(x_train, y_train, data_dir)
        write_json_list([train_idx.tolist(), test_idx.tolist()], os.path.join(data_dir, 'split_index.json'))
        generate_input_sequence_ete(x_train, y_train, data_dir, "train_sequences_{}.json".format(count),
                                    use_entropy=True, include_med=True)
        generate_input_sequence_ete(x_test, y_test, data_dir, "test_sequences_{}.json".format(count),
                                    use_entropy=True, include_med=True)
        # fast vs not fast
        for _type in ['fast', 'slow', 'unknown']:
            new_y_train = []
            if _type == 'unknown':
                for k in y_train:
                    if k == 'pos' or k == 'neg':
                        new_y_train.append('neg')
                    else:
                        new_y_train.append('pos')
            else:
                for k in y_train:
                    if k == 'neu':
                        if _type == 'fast':
                            new_y_train.append('neg')
                        else:
                            new_y_train.append('pos')
                    else:
                        new_y_train.append(k)
            # dataset_time_discretize(x_train, new_y_train, data_dir)

            generate_input_sequence_ete(x_train, new_y_train, data_dir, 'input_sequences_' + str(count) + '.json',
                                        use_entropy=True)
            if _type in ['fast', 'unknown']:
                CDSPM(data_dir, count, min_gr=1.5, file_predix=_type, pattern_type='pos')
            else:
                CDSPM(data_dir, count, min_gr=1.5, file_predix=_type, pattern_type='neg')

        test, pred, temp_list = validate_seq_vector_3(data_dir, count, use_csp=True, use_PCA=False)
        total_test_3 += test
        for i in pred:
            total_pred_3.append(i)

        # for _type in ['fast', 'slow']:
        #     test, pred, temp_list = validate_seq_vector_2(data_dir, count, use_csp=True, use_PCA=False, predix=_type)
        #     total_test_2[_type] += test
        #     for i in pred:
        #         total_pred_2[_type].append(i)

        count += 1

    # for _type in ['fast', 'slow']:
    #     print('------------{} vs not {}-------------'.format(_type, _type))
    #     print(confusion_matrix(total_test_2[_type], total_pred_2[_type], labels=['pos', 'neg']))
    #     print(classification_report(total_test_2[_type], total_pred_2[_type]))
    #     write_json_data(classification_report(total_test_2[_type], total_pred_2[_type], output_dict=True),
    #                     os.path.join(data_dir, 'classification_report_{}.json'.format(_type)))

    print('------------fast vs median vs slow-------------')
    print(confusion_matrix(total_test_3, total_pred_3, labels=['pos', 'neu', 'neg']))
    print(classification_report(total_test_3, total_pred_3))
    write_json_data(classification_report(total_test_3, total_pred_3, output_dict=True), os.path.join(data_dir, 'classification_report_3.json'))


def conduct_CDSPM(repo_name, max_t, min_len, max_len):
    model_sequence(repo_name)

    data_dir = get_global_val('result_dir')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    cut_sequence(min_len, max_len, max_t, repo_name, interval=3)

    X, Y, D = generate_dataset(repo_name)
    data_dir = os.path.join(get_global_val('result_dir'), "{}_{}_{}".format(repo_name, str(min_len), str(max_len)))
    # data_dir = os.path.join(get_global_val('result_dir'), "{}_{}_{}_paras".format(repo_name, str(min_len), str(max_len)))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # data_dir = os.path.join(data_dir, "sup{}_gr{}".format(0.1, 1.75))
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    write_json_list(D, os.path.join(data_dir, 'all_sequences.json'))

    # d = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=10)
    d = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)

    total_test = []
    total_pred = []
    count = 1
    for train_idx, test_idx in d.split(X, Y):
        x_train, x_test = numpy.array(X, dtype=object)[train_idx], numpy.array(X, dtype=object)[test_idx]
        y_train, y_test = numpy.array(Y, dtype=object)[train_idx], numpy.array(Y, dtype=object)[test_idx]

        write_json_list([train_idx.tolist(), test_idx.tolist()], os.path.join(data_dir, 'split_index.json'))

        dataset_time_discretize(x_train, y_train, data_dir)
        generate_input_sequence_ete(x_train, y_train, data_dir, "train_sequences_{}.json".format(count), use_entropy=True,
                                    include_med=False)
        generate_input_sequence_ete(x_test, y_test, data_dir, "test_sequences_{}.json".format(count), use_entropy=True,
                                    include_med=False)
        generate_input_sequence_ete(x_train, y_train, data_dir, 'input_sequences_' + str(count) + '.json',
                                    use_entropy=True)
        CDSPM(data_dir, count, min_gr=1.5)

        test, pred, temp_list = validate_seq_vector(data_dir, count, use_csp=True)
        total_test += test
        for i in pred:
            total_pred.append(i)
        count += 1

    print('------------thresholds = 60 days-------------')
    print(confusion_matrix(total_test, total_pred, labels=['pos', 'neg']))
    print(classification_report(total_test, total_pred))
    write_json_data(classification_report(total_test, total_pred, output_dict=True),
                    os.path.join(data_dir, 'classification_report.json'))


def conduct_CDSPM_total(repo_name, max_t, min_len, max_len):
    model_sequence(repo_name)
    cut_sequence(min_len, max_len, max_t, repo_name, interval=3)
    X, Y, D = generate_dataset(repo_name)
    data_dir = get_global_val('result_dir') + repo_name + '_'+str(min_len)+'_total'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    dataset_time_discretize(X, Y, data_dir)
    generate_input_sequence_ete(X, Y, data_dir, "train_sequences_0.json",
                                use_entropy=True, include_med=True)
    # generate_input_sequence_ete(X, Y, data_dir, 'input_sequences_0.json', use_entropy=True)
    for _type in ['fast', 'slow', 'unknown']:
        new_y = []
        if _type == 'unknown':
            for k in Y:
                if k == 'pos' or k == 'neg':
                    new_y.append('neg')
                else:
                    new_y.append('pos')
        else:
            for k in Y:
                if k == 'neu':
                    if _type == 'fast':
                        new_y.append('neg')
                    else:
                        new_y.append('pos')
                else:
                    new_y.append(k)
        # dataset_time_discretize(x_train, new_y_train, data_dir)

        generate_input_sequence_ete(X, new_y, data_dir, 'input_sequences_0.json',
                                    use_entropy=True)
        if _type in ['fast', 'unknown']:
            CDSPM(data_dir, 0, min_gr=1.5, file_predix=_type, pattern_type='pos')
        else:
            CDSPM(data_dir, 0, min_gr=1.5, file_predix=_type, pattern_type='neg')
        # CDSPM(data_dir, 0, min_gr=1.5, file_predix=_type)

    # CDSPM(data_dir, 0)


def conduct_CDSPM_ablation_e(repo_name, max_t, min_len, max_len):
    # suffix = '_e'
    suffix = ''
    cut_sequence(min_len, max_len, max_t, repo_name, interval=3, suffix=suffix)

    X, Y, D = generate_dataset(repo_name, suffix=suffix)
    data_dir = get_global_val('result_dir') + repo_name + '_' + str(min_len) + '_' + str(max_len) + '_e'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    write_json_list(D, os.path.join(data_dir, 'all_sequences.json'))
    d = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)

    total_test_3 = []
    total_pred_3 = []
    count = 1
    for train_idx, test_idx in d.split(X, Y):
        x_train, x_test = numpy.array(X, dtype=object)[train_idx], numpy.array(X, dtype=object)[test_idx]
        y_train, y_test = numpy.array(Y, dtype=object)[train_idx], numpy.array(Y, dtype=object)[test_idx]
        write_json_list([train_idx.tolist(), test_idx.tolist()], os.path.join(data_dir, 'split_index.json'))

        generate_input_sequence_e(x_train, y_train, data_dir, "train_sequences_{}.json".format(count),
                                  include_med=True)
        generate_input_sequence_e(x_test, y_test, data_dir, "test_sequences_{}.json".format(count),
                                  include_med=True)
        # fast vs not fast
        for _type in ['fast', 'slow', 'unknown']:
            new_y_train = []
            if _type == 'unknown':
                for k in y_train:
                    if k == 'pos' or k == 'neg':
                        new_y_train.append('neg')
                    else:
                        new_y_train.append('pos')
            else:
                for k in y_train:
                    if k == 'neu':
                        if _type == 'fast':
                            new_y_train.append('neg')
                        else:
                            new_y_train.append('pos')
                    else:
                        new_y_train.append(k)
            # dataset_time_discretize(x_train, new_y_train, data_dir)

            generate_input_sequence_e(x_train, new_y_train, data_dir, "input_sequences_{}.json".format(count),
                                      include_med=True)
            if _type in ['fast', 'unknown']:
                CDSPM(data_dir, count, min_gr=1.5, file_predix=_type, pattern_type='pos')
            else:
                CDSPM(data_dir, count, min_gr=1.5, file_predix=_type, pattern_type='neg')

        test, pred, temp_list = validate_seq_vector_3(data_dir, count, use_csp=True, use_PCA=True)
        total_test_3 += test
        for i in pred:
            total_pred_3.append(i)
        count += 1

    print('------------fast vs median vs slow-------------')
    print(confusion_matrix(total_test_3, total_pred_3, labels=['pos', 'neu', 'neg']))
    print(classification_report(total_test_3, total_pred_3))
    write_json_data(classification_report(total_test_3, total_pred_3, output_dict=True), os.path.join(data_dir, 'classification_report_3.json'))


if __name__ == '__main__':
    initialize()
    # select_issue_longer_than(min_len=min_len, repo_name=repo)
    # # calculate_fix_time(repo_name=repo, min_len=min_len)
    # t = classify_sequence(repo_name='tensorflow', len_=10)
    # model_sequence('tensorflow')
    for repo in ['flutter', 'react-native']:
        # t = classify_sequence(repo_name=repo, len_=10)
        # conduct_CDSPM_total(repo, 99999999, min_len=9, max_len=29)
        translate_result(repo)
    exit(-1)
    # extract_raw_data()
    for repo in ['tensorflow', 'go', 'rust', 'transformers', 'angular',
                 'flutter', 'rails', 'vscode', 'kubernetes',  'node',
                 'godot', 'react-native', 'fastlane', 'electron', 'core', 'pytorch', 'total']:
        if repo in ['flutter', 'react-native']:
            continue
        # translate_result(repo)
        # recommend_actions(repo, model_idx=1)
    # issue_preprocess(repo_name=repo)
    # delete_closed_by_bot(repo_name=repo)
    # data_dir = get_global_val('data_dir')+repo
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # filename = os.path.join(data_dir, 'event_id.json')
    # generate_event_id(repo, filename)

    # min_len = 10
    # max_len = 29
    # select_issue_longer_than(min_len=min_len, repo_name=repo)
    # calculate_fix_time(repo_name=repo, min_len=min_len)
    # t = classify_sequence(repo_name=repo, len_=min_len)
    # conduct_CDSPM_class3(repo, 99999999, min_len=min_len - 1, max_len=max_len)
    # exit(-1)
    # # data_dir = get_global_val('result_dir') + 'tensorflow_9_30_test/'
    # # test, pred, temp_seq = validate_seq_vector(data_dir, 1, use_PCA=True)
    # # select_issue_longer_than(min_len=min_len, repo_name=repo)
    # calculate_fix_time(repo_name=repo, min_len=min_len)
    # # t = classify_sequence(repo_name=repo, len_=min_len)
    # # conduct_CDSPM(repo, t, min_len=min_len - 1, max_len=max_len)
    # conduct_CDSPM_total(repo, 99999999, min_len=min_len - 1, max_len=30)
    # translate_result(repo)
    # for min_len in range(10, 11):
    #     # select_issue_longer_than(min_len=min_len, repo_name=repo)
    #     # calculate_fix_time(repo_name=repo, min_len=min_len)
    #     classify_sequence(repo_name=repo, len_=min_len)
    #     max_len = min_len + 15
    #     # conduct_CDSPM_total(repo, min_len=min_len-1, max_len=max_len)
    #     conduct_CDSPM(repo, min_len=min_len - 1, max_len=max_len)

    # for max_len in range(10, 31):
    #     conduct_CDSPM_ablation_e(repo, min_len=min_len - 1, max_len=max_len)
    #     conduct_CDSPM(repo, min_len=min_len - 1, max_len=max_len)
    # conduct_CDSPM_total(repo, min_len=min_len - 1, max_len=30)
    # conduct_CDSPM_ablation(repo, min_len=min_len - 1, max_len=30)



        # write_json_list(false_seq, os.path.join(get_global_val('result_dir'), repo_name+'_false_predicted_sequence_length.json'))

    # validate_seq_vector(get_global_val('result_dir') + '/entropy_16_26/')

    # translate_result(repo)

    # data_dir = os.path.join(get_global_val('data_dir'), 'godot')
    # pos = load_json_list(os.path.join(data_dir, 'issue_sequences_pos.json'))
    # neg = load_json_list(os.path.join(data_dir, 'issue_sequences_neg.json'))
    #
    # pos_id = set()
    # neg_id = set()
    # for i in pos:
    #     pos_id.add(i['_id'])
    # for i in neg:
    #     neg_id.add(i['_id'])
    #
    # data_dir = get_global_val('data_dir')
    # fix_time = load_json_data(os.path.join(data_dir, 'godot_closed_issues_len10_fix_time.json'))
    #
    # pos_time = []
    # neg_time = []
    #
    # for i in pos_id:
    #     pos_time.append(fix_time[i])
    # for i in neg_id:
    #     neg_time.append(fix_time[i])
    #
    # print(max(pos_time), min(pos_time))
    # print(max(neg_time), min(neg_time))
    # print(max(pos_time)/60/24, min(pos_time))
    # print(max(neg_time), min(neg_time)/60/24)
    # calcu_inconsistent_ratio()

    # show_event_freq()
    # show_event_interval()
    # sequence_length_show()


    # if_event = find_infrequent_event()
    # translate_sequences(['LockedEvent']+list(if_event))
    # load_event_log()
    # get_event_duration()
    # calculate_statics()
    # generate_graph()

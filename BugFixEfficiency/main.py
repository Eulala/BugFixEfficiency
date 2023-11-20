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


def conduct_CDSPM():
    # classify_sequence_by_close_interval('1day')
    # classify_sequence_by_close_time()
    # classify_issues()
    # classify_sequence()
    # model_sequence()
    # exit(-1)
    #
    data_dir = get_global_val('result_dir')
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # cut_sequence(15, interval=3)
    # exit(-1)

    # cut_sequence_by_time(_time=1, interval=3)
    # time_discretize(data_dir,  pair=True)

    X, Y, D = generate_dataset()
    data_dir = get_global_val('result_dir') + '/entropy_new/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    write_json_list(D, data_dir + 'all_sequences.json')

    # d = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=10)
    d = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)

    acc = []
    for train_idx, test_idx in d.split(X, Y):
        x_train, x_test = numpy.array(X, dtype=object)[train_idx], numpy.array(X, dtype=object)[test_idx]
        y_train, y_test = numpy.array(Y, dtype=object)[train_idx], numpy.array(Y, dtype=object)[test_idx]
        dataset_time_discretize(x_train, y_train, data_dir)
        write_json_list([train_idx.tolist(), test_idx.tolist()], os.path.join(data_dir, 'split_index.json'))

        data_dir = get_global_val('result_dir') + '/entropy_new/'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        generate_input_sequence_ete(x_train, y_train, data_dir, 'input_sequences.json', use_entropy=True)
        generate_input_sequence_ete(x_test, y_test, data_dir, 'test_sequences.json', use_entropy=True)
        generate_all_sequence_ete(D, data_dir, 'all_sequences_symbol_ver.json', use_entropy=True)

        CDSPM(data_dir)
        exit(-1)


        # a = validate_seq(data_dir)
        # acc.append(a)
        # exit(-1)
    # print(numpy.array(acc).mean())



    # data_dir = get_global_val('result_dir') + '/quartile/'
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # generate_input_sequence_ete(data_dir, use_entropy=False)  # for sequence mining
    # data_dir = get_global_val('result_dir') + '/quartile/'
    # CDSPM(data_dir)

    # data_dir = get_global_val('result_dir') + '/entropy/'
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # generate_input_sequence_ete(data_dir, use_entropy=True)
    #
    # data_dir = get_global_val('result_dir') + '/entropy/'
    # CDSPM(data_dir)
    #
    # length = 20
    # data_dir = get_global_val('result_dir') + 'event_time/len'+str(length)+'/'
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # cut_sequence(length, interval=2)
    # time_discretize(data_dir, suffix=str(length)+'_2', pair=False)
    # time_discretize_entropy_auto(data_dir, suffix=str(length)+'_2', pair=False)
    # data_dir = get_global_val('result_dir') + 'event_time/len'+str(length)+'/quartile/'
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # generate_input_sequence(data_dir, use_entropy=False, file_suffix='_'+str(length)+'_2')  # for sequence mining
    # data_dir = get_global_val('result_dir') + 'event_time/len'+str(length)+'/entropy/'
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # generate_input_sequence(data_dir, use_entropy=True, file_suffix='_'+str(length)+'_2')
    # data_dir = get_global_val('result_dir') + 'event_time/len'+str(length)+'/quartile/'
    # CDSPM(data_dir)
    # data_dir = get_global_val('result_dir') + 'event_time/len'+str(length)+'/entropy/'
    # CDSPM(data_dir)


if __name__ == '__main__':
    initialize()
    # extract_raw_data()
    # find_commit_repo()
    # commits = get_commit_list()
    # select_commits(commits)
    # select_closed_issue()

    # generate_commit_loc()
    # add_commitDiff_to_issues()

    # modify_pr_occur()
    # normalize_fix_time()
    # set_efficiency()
    # generate_sequence()
    # translate_sequences(['LockedEvent'])  # for process mining

    # issue_preprocess()
    # cat_comments_to_issues()
    # delete_less_than(length=21)
    # delete_faster_than()
    # calcu_close_time()
    # calcu_last_close_event('1day')
    # calcu_fix_time()
    # conduct_CDSPM()
    # validate_seq_vector(get_global_val('result_dir') + '/entropy_test/')

    recommend_actions()
    # translate_result()


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

import os
from common.io_util import *
from common.tagger_utils import span_iobes
from loader.instance import *
from loader.data_manager import DataManager
from loader.vocab_manager import VocabManager


def line_to_dic(line):
    arr = line.split('\t')
    name = arr[0]
    ele_dic = {}
    for ele in arr[1:]:
        ele_arr = ele.split(':')
        ele_dic[ele_arr[0]] = ele_arr[1]

    return name, ele_dic

def read_ranges(path):
    print('Read sent range from: ', path)
    sent_ranges = []
    for line in read_lines(path):
        arr = line.split()
        sent_ranges.append((int(arr[0]), int(arr[1])))
    return sent_ranges

def read_data(name):
    data_path = train_path if name == 'train' else dev_path if name == 'dev' else test_path
    data_range_path = train_range_path if name == 'train' else dev_range_path if name == 'dev' else test_range_path
    nlp_path = train_nlp_path if name == 'train' else dev_nlp_path if name == 'dev' else test_nlp_path

    sent_ranges = read_ranges(data_range_path)
    data_man = DataManager()

    nlp_lines = read_json_lines(nlp_path)

    pre_docname = None
    sent_num_in_doc = 0
    for i, lines in enumerate(read_multi_line_sent(data_path)):
        group_id = lines[0].split('\t')[1]
        doc_name = lines[1].split('\t')[1]
        sent = lines[2].split('\t')[1]
        words = sent.split()
        pos = lines[3].split('\t')[1]
        pos = pos.split()
        nlp_deps = nlp_lines[i]['nlp_deps']
        assert len(nlp_deps) == len(words)

        if doc_name == pre_docname:
            sent_num_in_doc += 1
        else:
            pre_docname = doc_name
            sent_num_in_doc = 0


        data_man.add_seq(SeqInst(group_id, str(sent_num_in_doc), words))
        data_man.add_single_label(SingleLabelInst(group_id, doc_name))
        data_man.add_pos(SeqInst(group_id, group_id, pos))

        sent_start, sent_end = sent_ranges[i]
        data_man.add_sent_range(SpanInst(group_id, str(sent_num_in_doc), start_loc=sent_start, end_loc=sent_end))
        data_man.add_dep(SeqInst(group_id, group_id, nlp_deps))

        ent_tag_tuples = []
        tri_tag_tuples = []

        for line in lines:
            if line.startswith('ENT'):
                name, ent_dic = line_to_dic(line)
                start, end = int(ent_dic['start']), int(ent_dic['end'])
                data_man.add_ent(SpanInst(group_id, ent_dic['id'], start, end ,
                                          ent_dic['coarse_type'], ent_dic['fine_grained_type'], ent_dic['ref_id']))

                ent_tag_tuples.append((start, end, ent_dic['coarse_type']))

            elif line.startswith('TRI'):
                name, tri_dic = line_to_dic(line)
                start, end = int(tri_dic['start']), int(tri_dic['end'])
                data_man.add_tri(SpanInst(group_id, tri_dic['id'], start, end, fine_grained_type=tri_dic['event_type']))
                tri_tag_tuples.append((start, end, tri_dic['event_type']))


            elif line.startswith('ARG'):
                name, arg_dic = line_to_dic(line)
                data_man.add_arg(ArgInst(group_id, arg_dic['id'], arg_dic['tri_id'], arg_dic['ent_id'], role_type=arg_dic['collapsed_argument_type']))

        ent_tags = span_iobes(len(words), ent_tag_tuples, typed=True)
        data_man.add_ent_tag_seq(SeqInst(group_id, group_id, ent_tags))

        tri_tags = span_iobes(len(words), tri_tag_tuples, typed=True)
        data_man.add_tri_tag_seq(SeqInst(group_id, group_id, tri_tags))

    return data_man

def read_pickle(root_path):
    vocab_pl_path = os.path.join(root_path, 'vocab.pkl')
    train_pl_path = os.path.join(root_path, 'train_event.pkl')
    dev_pl_path = os.path.join(root_path, 'dev_event.pkl')
    test_pl_path = os.path.join(root_path, 'test_event.pkl')

    vocab_man = VocabManager.load(vocab_pl_path)
    train_man = DataManager.load(train_pl_path)
    dev_man = DataManager.load(dev_pl_path)
    test_man = DataManager.load(test_pl_path)


    ent_group = train_man.group_ent_inst()
    tri_group = train_man.group_tri_inst()
    seq_to_remove = []
    word_num = 0
    for seq_inst in train_man.seq_inst_list:
        group_id = seq_inst.group_id
        if len(tri_group[group_id]) == 0 and len(ent_group[group_id]) < 3:
            seq_to_remove.append(seq_inst)
        else:
            sent_range = list(range(word_num, word_num + len(seq_inst)))
            word_num += len(seq_inst)
            seq_inst.add_value('sent_range', sent_range)

    for rm in seq_to_remove:
        train_man.seq_inst_list.remove(rm)
        train_man.group_ids.remove(rm.group_id)

    # dev sent range
    word_num = 0
    for seq_inst in dev_man.seq_inst_list:
        sent_range = list(range(word_num, word_num + len(seq_inst)))
        word_num += len(seq_inst)
        seq_inst.add_value('sent_range', sent_range)

    # test sent range
    word_num = 0
    for seq_inst in test_man.seq_inst_list:
        sent_range = list(range(word_num, word_num + len(seq_inst)))
        word_num += len(seq_inst)
        seq_inst.add_value('sent_range', sent_range)

    return vocab_man, train_man, dev_man, test_man

if __name__ == '__main__':
    root = '../../dataset/ACE2005/data_files/'
    opt_config = read_yaml('../../zjcexp/event_joint/opt_config.yaml')


    train_path = os.path.join(root, 'train.txt')
    dev_path = os.path.join(root, 'dev.txt')
    test_path = os.path.join(root, 'test.txt')

    train_range_path = os.path.join(root, 'arc_cache/ace_train_arc_range.txt')
    dev_range_path = os.path.join(root, 'arc_cache/ace_dev_arc_range.txt')
    test_range_path = os.path.join(root, 'arc_cache/ace_test_arc_range.txt')

    train_nlp_path = '/media/erwin/新加卷1/workspace/NLP/transition_event/data_files/ace05_en_fullinfo/train_nlp.json'
    dev_nlp_path = '/media/erwin/新加卷1/workspace/NLP/transition_event/data_files/ace05_en_fullinfo/dev_nlp.json'
    test_nlp_path = '/media/erwin/新加卷1/workspace/NLP/transition_event/data_files/ace05_en_fullinfo/test_nlp.json'


    vocab_pl_path = os.path.join(root, 'vocab.pkl')
    vocab_txt_path = os.path.join(root, 'vocab.txt')
    train_pl_path = os.path.join(root, 'train_event.pkl')
    dev_pl_path = os.path.join(root, 'dev_event.pkl')
    test_pl_path = os.path.join(root, 'test_event.pkl')


    train_man = read_data('train')
    dev_man = read_data('dev')
    test_man = read_data('test')
    #data_man.show_statistic()
    vocab_man = VocabManager()
    vocab_man.build_vocab(train_man)
    vocab_man.update_vocab(dev_man)
    vocab_man.update_vocab(test_man)

    vocab_man.to_indices(train_man)
    vocab_man.to_indices(dev_man)
    vocab_man.to_indices(test_man)

    vocab_man.cache_pretrained_word_embedding(opt_config['embedding_dir'], opt_config['embedding_type'],
                                              opt_config['embedding_file'], opt_config['vec_npy_file'])

    train_man.save(train_pl_path)
    dev_man.save(dev_pl_path)
    test_man.save(test_pl_path)

    vocab_man.save(vocab_pl_path)

    vocab_man.write_text(root)

    vocab_man, train_man, dev_man, test_man = read_pickle(root)

    print(train_man.group_size())

import numpy as np
import torch
import torch.nn as nn
import layer
from common.pad_utils import pad_and_mask, pad_sequence, pad_3d
from common.tensor_utils import  batched_index_select, expand_dim_to
from layer.loss import sequence_cross_entropy_with_logits
from loader.vocab import Vocab
import random

class Network(torch.nn.Module):

    def __init__(self, opt_config, word_vec, vocab_man, train_sent_arr=None, dev_sent_arr=None, test_sent_arr=None,
                 train_arc_arr=None, dev_arc_arr=None, test_arc_arr=None):
        super(Network, self).__init__()
        self.opt_config = opt_config
        self.device = opt_config['device']
        self.vocab_man = vocab_man
        self.ent_tag_start = self.vocab_man.ent_tag_vocab.get_index(Vocab.NULL)
        self.ent_tag_pad = self.vocab_man.ent_tag_vocab.get_index(Vocab.PAD)

        self.train_arc_arr = train_arc_arr
        self.dev_arc_arr = dev_arc_arr
        self.test_arc_arr = test_arc_arr

        if self.opt_config['use_sentence_vec']:
            self.train_sent_arr = train_sent_arr
            self.dev_sent_arr = dev_sent_arr
            self.test_sent_arr = test_sent_arr

        self.word_embedding = layer.Embedding(opt_config['n_words'], opt_config['word_embed_dim'], pretrained_vec=word_vec)
        self.pos_embedding = layer.Embedding(opt_config['n_pos'], opt_config['pos_embed_dim'])
        self.event_embedding = layer.Embedding(opt_config['n_event_type'], opt_config['tri_emb_dim'])
        self.entity_embedding = layer.Embedding(opt_config['n_ent_type'], opt_config['ent_emb_dim'])
        self.ent_tag_embedding = layer.Embedding(opt_config['n_ent_tag_type'], opt_config['ent_tag_emb_dim'])

        self.dp_emb = nn.Dropout(opt_config['dp_emb'])

        rnn_input_dim = opt_config['word_embed_dim']
        if opt_config['use_pos']:
            rnn_input_dim += opt_config['pos_embed_dim']

        if self.opt_config['use_sentence_vec']:
            rnn_input_dim += opt_config['sent_vec_dim']


        bi_rnn_dim = self.opt_config['rnn_dim'] * 2
        label_emb_dim = bi_rnn_dim #160

        self.rnn = layer.VarRNN(rnn_input_dim, self.opt_config['rnn_dim'], n_layer=self.opt_config['rnn_layer'],
                           bidirectional=True, drop_out=self.opt_config['dp_rnn'])

        self.rnn_layer2 = layer.VarRNN(bi_rnn_dim + label_emb_dim, self.opt_config['rnn_dim'], n_layer=self.opt_config['rnn_layer'],
                                bidirectional=True, drop_out=self.opt_config['dp_rnn'])
        self.rnn_dp = nn.Dropout(self.opt_config['dp_rnn'])

        self.rnn_ent = layer.VarRNN(rnn_input_dim, self.opt_config['rnn_dim'], n_layer=self.opt_config['rnn_layer'], bidirectional=True, drop_out=self.opt_config['dp_rnn'])
        self.rnn_tri = layer.VarRNN(rnn_input_dim, self.opt_config['rnn_dim'], n_layer=self.opt_config['rnn_layer'], bidirectional=True, drop_out=self.opt_config['dp_rnn'])
        self.rnn_arg = layer.VarRNN(rnn_input_dim, self.opt_config['rnn_dim'], n_layer=self.opt_config['rnn_layer'], bidirectional=True, drop_out=self.opt_config['dp_rnn'])


        self.ent_tag_out = nn.Sequential(
            nn.Linear(bi_rnn_dim + opt_config['ent_tag_emb_dim'], label_emb_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
            #nn.Linear(self.opt_config['tri_tag_hidden_dim'], self.opt_config['n_ent_tag_type'])
        )

        self.tri_tag_out = nn.Sequential(
            nn.Linear(bi_rnn_dim, label_emb_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(self.opt_config['tri_tag_hidden_dim'], self.opt_config['n_tri_tag'])
        )

        self.arg_out = nn.Sequential(
            nn.Linear(bi_rnn_dim * 2 + opt_config['tri_emb_dim'] + opt_config['ent_emb_dim'], #+ opt_config['n_tri_tag'],
                      label_emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            #nn.Linear(self.opt_config['arg_hidden_dim'], self.opt_config['n_arg_types'])
        )

        #self.joint_label_embedding = nn.Linear(label_emb_dim,opt_config['n_ent_tag_type'] + opt_config['n_tri_tag'] + opt_config['n_arg_types'],bias=False)

        # joint_label_embedding
        self.ent_mat = nn.Linear(label_emb_dim, opt_config['n_ent_tag_type'], bias=True)
        self.tri_mat = nn.Linear(label_emb_dim, opt_config['n_tri_tag'], bias=True)
        self.arg_mat = nn.Linear(label_emb_dim, opt_config['n_arg_types'], bias=True)

        num_heads = 4
        self.multi_head = layer.MultiheadAttention(label_emb_dim, num_heads)

        self.soft_max = nn.Softmax(dim=-1)

    def joint_output(self, ent_tag_reps, tri_tag_reps, arg_seq_reps):
        batch_size, seq_len, _ = ent_tag_reps.shape
        _, arg_seq_len, _ = arg_seq_reps.shape

        ent_tag_score = self.ent_mat(ent_tag_reps)

        tri_tag_score = self.tri_mat(tri_tag_reps)

        arg_seq_score = self.arg_mat(arg_seq_reps)


        return ent_tag_score, tri_tag_score, arg_seq_score

    def forward(self, input_dic, type='train'):
        seq_mask = input_dic['seq_mask']
        batch_size, max_seq_len = seq_mask.shape
        rnn_share, rnn_ent, rnn_tri, rnn_arg = self.encoder(input_dic, type)
        # rnn_cat_ent = torch.cat([rnn_share, rnn_ent], dim=-1)
        # rnn_cat_tri = torch.cat([rnn_share, rnn_tri], dim=-1)
        # rnn_cat_arg = torch.cat([rnn_share, rnn_arg], dim=-1)


        ent_tag_previous = input_dic['ent_tag_previous']
        ent_tag_pre_emb = self.ent_tag_embedding(ent_tag_previous)
        ent_tag_input = torch.cat([rnn_share, ent_tag_pre_emb], dim=-1)
        # (batch_size, seq_len, label_emb_dim)
        ent_tag_reps = self.ent_tag_out(ent_tag_input)

        ent_tag_gold = input_dic['ent_tag']

        #rnn_cat_tri = torch.cat([rnn_cat_tri, self.soft_max(ent_tag_score)], dim=-1)
        tri_tag_reps = self.tri_tag_out(rnn_share)

        tri_tag_gold = input_dic['tri_tag']

        # (batch_size, max_arg_seq_len)
        arg_role_type = input_dic['arg_role_type']
        # (batch_size, max_arg_seq_len)
        arg_seq_mask = input_dic['arg_seq_mask']


        # (batch_size, max_arg_seq_len, n_arg_types)
        arg_seq_reps = self.argument_forward(rnn_share, input_dic)

        ent_tag_score, tri_tag_score, arg_seq_score = self.joint_output(ent_tag_reps, tri_tag_reps, arg_seq_reps)

        ent_tag_loss = sequence_cross_entropy_with_logits(ent_tag_score, ent_tag_gold, seq_mask,
                                                          batch_average=True, batch_seq_average=True)

        tri_tag_loss = sequence_cross_entropy_with_logits(tri_tag_score, tri_tag_gold, seq_mask,
                                                          batch_average=True, batch_seq_average=True)

        arg_loss = sequence_cross_entropy_with_logits(arg_seq_score, arg_role_type, arg_seq_mask,
                                                          batch_average=True, batch_seq_average=True, handle_nan=False)
        #arg_loss = arg_loss.sum() / arg_seq_mask.sum()

        return ent_tag_loss, tri_tag_loss, arg_loss



    def argument_forward(self, reps, input_dic):
        # (batch_size, max_arg_seq_len)
        arg_tri_loc = input_dic['arg_tri_loc']
        # (batch_size, max_arg_seq_len)
        arg_ent_loc = input_dic['arg_ent_loc']
        # (batch_size, max_ent_size, max_ent_len)
        ent_tensor_3d = input_dic['ent']
        # (batch_size, max_ent_size, max_ent_len)
        ent_mask_3d = input_dic['ent_mask']

        # (batch_size, max_arg_seq_len)
        arg_tri_type = input_dic['arg_tri_type']
        arg_tri_type_emb = self.event_embedding(arg_tri_type)
        # (batch_size, max_arg_seq_len)
        arg_ent_type = input_dic['arg_ent_type']
        arg_ent_type_emb = self.entity_embedding(arg_ent_type)

        # (batch_size, max_arg_seq_len, fea_dim)
        arg_tri_reps = batched_index_select(reps, arg_tri_loc)
        #tri_tag_probs = self.soft_max(tri_tag_score)
        #arg_tri_tag_prob = batched_index_select(tri_tag_score, arg_tri_loc)

        # (batch_size, max_ent_size, max_ent_len, fea_dim)
        ent_reps = batched_index_select(reps, ent_tensor_3d)
        ent_mask_float = ent_mask_3d.float().sum(-1, keepdim=True)
        # avoid Nan
        ent_mask_float[ ent_mask_float == 0 ] = 1
        # (batch_size, max_ent_size, fea_dim)
        ent_reps_avg = ent_reps.sum(2) / ent_mask_float

        # (batch_size, max_arg_seq_len, fea_dim)
        arg_ent_reps = batched_index_select(ent_reps_avg, arg_ent_loc)

        # (batch_size, max_arg_seq_len, fea_dim * 2)
        arg_seq_reps = torch.cat([arg_tri_reps, arg_ent_reps, arg_tri_type_emb, arg_ent_type_emb], dim=-1)

        # (batch_size, max_arg_seq_len, n_arg_types)
        arg_seq_reps = self.arg_out(arg_seq_reps)

        return arg_seq_reps


    def encoder(self, input_dic, type):
        seq_tensor = input_dic['seq']
        seq_mask = input_dic['seq_mask']
        seq_lens = input_dic['seq_lens']
        word_emb = self.word_embedding(seq_tensor)

        if self.opt_config['use_pos']:
            pos_emb = self.pos_embedding(input_dic['pos'])
            word_emb = torch.cat([word_emb, pos_emb], dim=-1)

        if self.opt_config['use_sentence_vec']:
            seq_range_tensor = input_dic['seq_range']
            sent_mat = self.train_sent_arr if type=='train' else self.dev_sent_arr if type=='dev' else self.test_sent_arr
            seq_range_shape = seq_range_tensor.shape
            sent_emb = torch.index_select(sent_mat, 0, seq_range_tensor.view(-1))
            sent_emb = sent_emb.reshape(*seq_range_shape, -1)
            word_emb = torch.cat([word_emb, sent_emb], dim=-1)

        word_emb = self.dp_emb(word_emb)
        rnn_share = self.rnn(word_emb, seq_mask)
        rnn_ent = self.rnn_ent(word_emb, seq_mask)
        rnn_tri = self.rnn_ent(word_emb, seq_mask)
        rnn_arg = self.rnn_ent(word_emb, seq_mask)

        joint_label_mat = torch.cat([self.tri_mat.weight], dim=0)
        joint_label_mat = joint_label_mat.transpose(0, 1).contiguous()

        label_attn_reps = self.multi_head(rnn_share, joint_label_mat, joint_label_mat)
        rnn_input = torch.cat([rnn_share, label_attn_reps], dim=-1)
        rnn_share = self.rnn_layer2(rnn_input, seq_mask) + rnn_share

        return rnn_share, rnn_ent, rnn_tri, rnn_arg


    def decode_ent(self, rnn_out, input_dic):
        seq_lens = input_dic['seq_lens']
        batch_size, seq_len, _ = rnn_out.shape
        batch_ent_tag_predict = []
        batch_ent_tag_pre = torch.tensor([self.ent_tag_start] * batch_size, dtype=torch.long, device=self.device)
        # (batch_size, ent_tag_dim)
        batch_ent_tag_emb = self.ent_tag_embedding(batch_ent_tag_pre)
        ent_tag_emb_list = []
        ent_out_list = []
        for step in range(seq_len):
            # (batch_size, rnn_dim * 2)
            rnn_step = rnn_out[:, step, :]
            ent_input = torch.cat([rnn_step, batch_ent_tag_emb], dim=-1)
            # (batch_size, n_ent_tag)
            ent_reps = self.ent_tag_out(ent_input)
            ent_out = self.ent_mat(ent_reps)
            ent_out_list.append(ent_out)
            batch_ent_tag_pre = torch.argmax(ent_out, -1, keepdim=False)
            batch_ent_tag_predict.append(batch_ent_tag_pre.tolist())

            batch_ent_tag_emb = self.ent_tag_embedding(batch_ent_tag_pre)
            ent_tag_emb_list.append(batch_ent_tag_emb)

        ent_tag_pred_score = torch.stack(ent_out_list, dim=1)

        # (seq_len, batch_size) -> (batch_size, seq_len)
        ent_tag_preds = np.array(batch_ent_tag_predict).T.tolist()
        #ent_tag_preds = input_dic['batch_ent_tags']
        # entity tags to span
        assert len(ent_tag_preds) == len(seq_lens)
        batch_pred_ents = []
        for tag_seq, seq_len in zip(ent_tag_preds, seq_lens):
            pred_tags, pred_types = [], []
            for tag in tag_seq[:seq_len]:
                tag_str = self.vocab_man.ent_tag_vocab.get_token(tag)
                tag_arr = tag_str.split('_')
                if len(tag_arr) == 1:
                    pred_tags.append('O')
                    pred_types.append(self.ent_tag_pad)
                else:
                    pred_tags.append(tag_arr[0])
                    pred_types.append(self.vocab_man.ent_coarse_type_vocab.get_index(tag_arr[1]))
            pred_ents = self.predict_ent(pred_tags, pred_types)
            batch_pred_ents.append(pred_ents)

        ent_list_3d = []
        for pred_ents in batch_pred_ents:
            sent_span_list = []
            for span in pred_ents:
                start, end, ent_type = span
                sent_span_list.append(list(range(start, end + 1)))
            ent_list_3d.append(sent_span_list)

        has_ent = any(ent_list_3d)
        if has_ent:
            ent_tensor_3d, ent_mask_3d = pad_3d(ent_list_3d)
            input_dic['ent'] = ent_tensor_3d.to(self.device)
            # (batch_size, max_ent_size, max_ent_len)
            input_dic['ent_mask'] = ent_mask_3d.to(self.device)


        return ent_tag_pred_score, batch_pred_ents, has_ent


    def decode_tri(self, rnn_out, ent_tag_score):
        #rnn_out = torch.cat([rnn_out, self.soft_max(ent_tag_score)], dim=-1)
        batch_size, seq_len, _ = rnn_out.shape


        tri_tag_reps = self.tri_tag_out(rnn_out)
        tri_tag_score = self.tri_mat(tri_tag_reps)

        max_idx = torch.argmax(tri_tag_score, dim=-1)
        tri_tag_pred = max_idx.detach().cpu().tolist()
        return tri_tag_score, tri_tag_pred


    def decode(self, input_dic, type='test'):
        seq_lens = input_dic['seq_lens']
        batch_size = len(seq_lens)
        with torch.no_grad():
            rnn_share, rnn_ent, rnn_tri, rnn_arg = self.encoder(input_dic, type)
            # rnn_cat_ent = torch.cat([rnn_share, rnn_ent], dim=-1)
            # rnn_cat_tri = torch.cat([rnn_share, rnn_tri], dim=-1)
            # rnn_cat_arg = torch.cat([rnn_share, rnn_arg], dim=-1)

            ent_tag_score, batch_pred_ents, has_ent = self.decode_ent(rnn_share, input_dic)
            #batch_gold_ents = input_dic['batch_ents_gold']

            tri_tag_score, tri_tag_pred = self.decode_tri(rnn_share, ent_tag_score)

            batch_arg_candis, arg_seq_lens, has_tri = self.construct_batch_arg_seq(tri_tag_pred, batch_pred_ents, input_dic)

            if (not has_tri) or (not has_ent):
                batch_arg_preds = [[] for _ in range(batch_size)]
                return batch_pred_ents, tri_tag_pred, batch_arg_preds

            arg_seq_reps = self.argument_forward(rnn_share, input_dic)
            arg_seq_out  = self.arg_mat(arg_seq_reps)


            # predictions .........
            try:
                max_idx = torch.argmax(arg_seq_out, dim=-1)
                batch_arg_type_pred = max_idx.detach().cpu().tolist()
            except RuntimeError:
                print('RuntimeError: CUDA error: device-side assert triggered')
                print(arg_seq_out)
                batch_arg_preds = [[] for _ in range(batch_size)]
                return batch_pred_ents, tri_tag_pred, batch_arg_preds

            batch_arg_preds = []
            for arg_candi_sent, arg_type_pred_sent, arg_seq_len in zip(batch_arg_candis, batch_arg_type_pred, arg_seq_lens):
                arg_preds = []
                for arg_pred_candi, arg_type_pred in zip(arg_candi_sent[:arg_seq_len], arg_type_pred_sent[:arg_seq_len]):
                    if arg_type_pred != self.opt_config['arg_null_type']:
                        arg_pred_candi.append(arg_type_pred)
                        arg_preds.append(arg_pred_candi)
                batch_arg_preds.append(arg_preds)

            return batch_pred_ents, tri_tag_pred, batch_arg_preds



    def construct_batch_arg_seq(self, batch_tri_tag_pred, batch_pred_ents, input_dic):
        seq_lens = input_dic['seq_lens']
        #batch_ent = input_dic['batch_ent']
        batch_arg_tri_types = []
        batch_arg_tri_locations = []  # location in the sentence
        batch_arg_ent_types = []
        batch_arg_ent_locations = []  # location in the selected list
        batch_arg_pred_tup = []
        has_element = False # avoid empty argument prediction
        for bnum, tri_tag_pred in enumerate(batch_tri_tag_pred):
            seq_len = seq_lens[bnum]
            arg_tri_types = []
            tri_locations = []  # location in the sentence
            arg_ent_type = []
            ent_locations = []  # location in the selected list
            arg_pred_tup = [] # (tri, ent)
            for i, tag in enumerate(tri_tag_pred[:seq_len]):
                if tag != self.opt_config['tri_tag_O']:
                    for ent_num, ent_pred_tup in enumerate(batch_pred_ents[bnum]):
                        ent_start, ent_end, ent_type = ent_pred_tup
                        tri_locations.append(i)
                        arg_tri_types.append(self.trigger_tag_to_event_type(tag))
                        ent_locations.append(ent_num)
                        arg_ent_type.append(ent_type)
                        # for evaluation (tri_loc_in_sent, ent_loc_in_sent, role_type)
                        arg_pred_tup.append([ent_start, ent_end, i])
                        has_element = True

            batch_arg_tri_locations.append(tri_locations)
            batch_arg_ent_locations.append(ent_locations)
            batch_arg_tri_types.append(arg_tri_types)
            batch_arg_ent_types.append(arg_ent_type)
            batch_arg_pred_tup.append(arg_pred_tup)

        arg_tri_loc, arg_seq_mask, arg_seq_lens = pad_and_mask(batch_arg_tri_locations)
        arg_ent_loc, _, _ = pad_and_mask(batch_arg_ent_locations)
        arg_tri_type, _, _ = pad_and_mask(batch_arg_tri_types)
        arg_ent_type, _, _ = pad_and_mask(batch_arg_ent_types)

        input_dic['arg_seq_mask'] = arg_seq_mask.to(self.device)
        input_dic['arg_tri_loc'] = arg_tri_loc.to(self.device)
        input_dic['arg_ent_loc'] = arg_ent_loc.to(self.device)

        input_dic['arg_tri_type'] = arg_tri_type.to(self.device)
        input_dic['arg_ent_type'] = arg_ent_type.to(self.device)

        return batch_arg_pred_tup, arg_seq_lens, has_element


    def trigger_tag_to_event_type(self, tag_idx):
        tag_str = self.vocab_man.tri_tag_vocab.get_token(tag_idx)
        event_type_idx = self.vocab_man.tri_vocab.get_index(tag_str[2:]) # cut first two chars i.e. B_XXX to XXX
        return event_type_idx


    def predict_ent(self, pred_tags, pred_types):
        pred_ents = set()
        ent_start, ent_end = -1, -1
        ent_type = -1
        for i, tag in enumerate(pred_tags):
            if tag == 'S':
                pred_ents.add((i, i, pred_types[i]))
                # pred_ents.add((i, i))
            elif tag == 'B':
                ent_start = i
                ent_type = pred_types[i]
            elif tag == 'E' or (ent_start != -1 and (i == len(pred_tags) - 1 or pred_tags[i + 1] == 'O')):
                ent_end = i
                #pred_ents.add((ent_start, ent_end, ent_type)) # (ent_head, ent_tail, ent_label)
                # end first
                if ent_start != -1 and ent_type != -1:
                    pred_ents.add((ent_start, ent_end, ent_type)) # (ent_tail, ent_head, ent_label)
                # pred_ents.add((ent_start, ent_end))
                ent_start, ent_end = -1, -1
                ent_type = -1
        return pred_ents


import copy
import sys
import torch
import pickle
import random
import numpy as np

from tqdm import tqdm
from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths

class Dataset(object):
    def __init__(self, batch_size, seq_len, dataset):
        super().__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.construct_index(dataset)

    def construct_index(self, dataset):
        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        return tqdm(self.reader(device, shuffle), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        if shuffle:
            self.shuffle()
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):
        # doc_id, role_name, input_ids, input_mask, segment_ids, doc_offset, token_to_orig_map, start_position, end_position
        sentence_lens = [len(data[2]) for data in batch]
        max_sentence_len = self.seq_len

        data_x, data_x_mask, data_segment_id, data_start, data_end = list(), list(), list(), list(), list()
        dep_matrix = list()
        appendix = list()

        for data in batch:
            doc_id, role_name, input_ids, input_mask, segment_ids, doc_offset, token_to_orig_map, start_position, end_position, deps, heads = data
            data_x.append(input_ids)
            data_x_mask.append(input_mask)
            data_segment_id.append(segment_ids)
            data_start.append(start_position)
            data_end.append(end_position)

            temp = np.zeros((max_sentence_len, max_sentence_len))
            for i in range(max_sentence_len):
                temp[i][i] = 1
            # for i, j in enumerate(deps):
            #     temp[i][j] = 0.005
            #     temp[j][i] = 0.005
            
            dep_matrix.append(temp)
            appendix.append([doc_id, role_name, doc_offset, token_to_orig_map])

        f = torch.LongTensor

        data_x = f(data_x)
        data_x_mask = f(data_x_mask)
        data_segment_id = f(data_segment_id)
        data_start = f(data_start)
        data_end = f(data_end)

        dep_matrix = np.asarray(dep_matrix)
        dep_matrix = torch.FloatTensor(dep_matrix)

        return [data_x.to(device),  
                data_x_mask.to(device),
                data_segment_id.to(device),
                data_start.to(device),
                data_end.to(device),
                dep_matrix.to(device),
                appendix]


if __name__ == '__main__':
    
    filename = 'data.pk'
    data = pickle.load(open(filename, 'rb'))
    train_data, dev_data, test_data = data

    train_set = Dataset(20, 512, train_data)

    for batch in train_set.reader('cpu', False):
        data_x, data_x_mask, data_segment_id, data_start, data_end, dep_matrix, appendix = batch
        print(data_x)
        break
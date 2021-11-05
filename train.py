import pickle
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from model import BertForQuestionAnswering
from transformers import AdamW, get_linear_schedule_with_warmup

import config

from dataset import Dataset
from utils import save_model, load_model

def build_model(model_path):
    model = BertForQuestionAnswering.from_pretrained(model_path)
    return model

def load_dataset():
    filename = 'data.pk'
    data = pickle.load(open(filename, 'rb'))
    train_data, dev_data, test_data = data
    return train_data, dev_data, test_data


if __name__ == '__main__':

    device = 'cuda:0'
    
    n_epoch = 10
    batch_size = 5

    train_data, dev_data, test_data = load_dataset()

    n_train_data = len(train_data)
    print('#Train', n_train_data)
    train_data = train_data[:int(n_train_data/100)]

    train_set = Dataset(batch_size, 512, train_data)
    test_set = Dataset(batch_size, 512, test_data)

    lr = 2e-5
    max_grad_norm = 1.0
    num_warmup_steps = 0
    num_training_steps = n_epoch * (len(train_data) / batch_size)
    warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

    model = build_model(config.bert_dir)
    # model = build_model('/home/jliu/data/BertModel/bert-base-uncased-squad')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

    idx = 0
    for _ in range(n_epoch):
        for batch in train_set.get_tqdm(device, True):
            idx += 1
            model.train()
            data_x, data_x_mask, data_segment_id, data_start, data_end, dep_matrix, appendix = batch

            inputs = {
                'input_ids': data_x,
                'attention_mask':  data_x_mask,
                'token_type_ids':  data_segment_id,
                'start_positions': data_start,
                'end_positions':   data_end,
                'dep_matrix': dep_matrix}
            outputs = model(**inputs)
            loss = outputs[0]
            loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        #save_model(model, 'models/%d.pk' % (idx))
        save_model(model, 'models/pre_trained_%d.pk' % (idx))
        from evaluate import evaluate
        evaluate(model, device, test_set, 'data/test.jsonlines')


import pickle
import torch
import os
import numpy as np
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from model import BertForQuestionAnswering
from transformers import AdamW, get_linear_schedule_with_warmup

import config

from preprocess import read_corpus
from dataset import Dataset
from utils import load_model

# https://nlp.jhu.edu/rams/

def build_model(model_path):
    model = BertForQuestionAnswering.from_pretrained(model_path)
    return model


def load_dataset():
    filename = 'data.pk'
    data = pickle.load(open(filename, 'rb'))
    train_data, dev_data, test_data = data
    return train_data, dev_data, test_data


def _get_target_index(token_to_orig_map, idx):
    result = []
    for i in token_to_orig_map:
        if token_to_orig_map[i] == idx:
            result.append(i)
    return result


def evaluate(model, device, data_set, file_name):
    fileout = open('eval.txt', 'w')

    test_data = read_corpus(file_name)
    golden_map = {}
    for elem in test_data:
        doc_id, sentences, trigger, entities, args = elem
        golden_map[doc_id] = [sentences, trigger, entities, args]
    
    predicted = predict(model, device, data_set)
    result = []
    for doc_id in golden_map:
        sentences, trigger, entities, args = golden_map[doc_id]
        trigger_s, trigger_e, trigger_type = trigger

        predicted_results = predicted.get(doc_id, list())  
        role_predictions = []
        for elem in predicted_results: # elem = s_prob, e_prob, role_name, token_to_orig_map
            s_prob, e_prob, role_name, token_to_orig_map = elem

            if True:
                s_max = np.argmax(s_prob)  ### Generate only one entity
                e_max = np.argmax(e_prob)
                
                found = False
                if s_max in token_to_orig_map and e_max in token_to_orig_map and s_max <= e_max and e_max-s_max<10:
                    s = token_to_orig_map[s_max]
                    e = token_to_orig_map[e_max]
                    found = True
                if found:
                    role_predictions.append([s, e, role_name])
            else:
                for entity in entities:
                    entity_s, entity_e = entity
                    s_index = _get_target_index(token_to_orig_map, entity_s)
                    e_index = _get_target_index(token_to_orig_map, entity_e)

                    s_min = min(s_index)
                    e_max = max(e_index)

                    probability = sum([s_prob[x] for x in range(s_min, e_max + 1)] + [e_prob[x] for x in range(s_min, e_max + 1)])
                    if probability > 0.08:
                        role_predictions.append([entity_s, entity_e, role_name])
                        break

        for elem in role_predictions: # [[57, 58, 'communicator']]
            elem.append(1.0)
        
        json_output = [ [trigger_s, trigger_e] ] + role_predictions
        json_output = [json_output]
        temp = {
            'doc_key': doc_id,
            'predictions': json_output
        }
        json.dump(temp, fileout)
        fileout.write('\n')
    fileout.close()

    os.system('python data/scorer/scorer.py --gold_file %s --pred_file eval.txt --ontology_file data/scorer/event_role_multiplicities.txt --do_all' % (file_name))



def predict(model, device, data_set):
    model.eval()

    predicted_s = []
    predicted_e = []
    appendixes = []

    for batch in data_set.get_tqdm(device, True):
        data_x, data_x_mask, data_segment_id, data_start, data_end, dep_matrix, appendix = batch

        inputs = {  'input_ids': data_x,
                    'attention_mask':  data_x_mask,
                    'token_type_ids':  data_segment_id,
                    'dep_matrix': dep_matrix}
        outputs = model(**inputs)
        
        predicted_s.extend(torch.softmax(outputs[0], -1).detach().cpu().numpy())
        predicted_e.extend(torch.softmax(outputs[1], -1).detach().cpu().numpy())
        appendixes.extend(appendix)

        # print(outputs[0][0])
    
    
    predicted_map = {}
    for app, s_prob, e_prob in zip(appendixes, predicted_s, predicted_e):

        doc_id, role_name, doc_offset, token_to_orig_map = app
        predicted_map.setdefault(doc_id, list())

        # s_max = np.argmax(s_prob)  ### Generate Multiple Answers
        # e_max = np.argmax(e_prob)
        
        # found = False
        # if s_max in token_to_orig_map and e_max in token_to_orig_map and s_max <= e_max and e_max-s_max<10:
        #     s = token_to_orig_map[s_max]
        #     e = token_to_orig_map[e_max]
        #     found = True
        # if found:
        #     predicted_map[doc_id].append([s, e, role_name])

        predicted_map[doc_id].append([s_prob, e_prob, role_name, token_to_orig_map])

    return predicted_map



if __name__ == '__main__':
    device = 'cuda:0'
    batch_size = 10 # 2, 5, 10, 20, 30

    train_data, dev_data, test_data = load_dataset()

    # filename = 'data_test_bt.pk'
    # test_data = pickle.load(open(filename, 'rb'))

    test_set = Dataset(batch_size, 512, test_data)
    dev_set = Dataset(batch_size, 512, dev_data)

    pre_train_dir = '../bert-base-uncased-squad'
    model = build_model(pre_train_dir)
    load_model(model, 'models/pre_trained_11114.pk')
    model.to(device)

    evaluate(model, device, dev_set, 'data/dev.jsonlines')
    evaluate(model, device, test_set, 'data/test.jsonlines')


    # python data/scorer/scorer.py --gold_file data/test.jsonlines --pred_file eval.txt --ontology_file data/scorer/event_role_multiplicities.txt --do_all\
    # python data/scorer/scorer.py --gold_file data/dev.jsonlines --pred_file eval.txt --ontology_file data/scorer/event_role_multiplicities.txt --do_all
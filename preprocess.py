import json
import pickle
import config
import spacy
from transformers import BertTokenizer

from utils import get_dep_and_head

tokenizer = BertTokenizer.from_pretrained(config.bert_dir)

def read_corpus(filename):
    results = []
    for line in open(filename):
        json_dict = json.loads(line)

        doc_id = json_dict['doc_key']
        sentences = json_dict['sentences']

        trigger = json_dict['evt_triggers'][0]
        trigger_s, trigger_e, trigger_type = trigger[0], trigger[1], trigger[-1][0][0]
        trigger = [trigger_s, trigger_e, trigger_type]

        entities = json_dict['ent_spans']
        entities = [[x[0], x[1]] for x in entities]

        args = json_dict['gold_evt_links']
        args = [[x[1][0], x[1][1], x[2]] for x in args]

        temp = [doc_id, sentences, trigger, entities, args]
        results.append(temp)
    return results


def _find_idx_from_dic(org_list, org_map):
    result = []
    for elem in org_list:
        for i in org_map:
            if org_map[i] == elem:
                result.append(i)
                break
    return result


def build_bert_example(query, context, trigger_s, trigger_e, start_pos, end_pos, max_seq_length,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0, 
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    cls_token_segment_id=0, pad_token_segment_id=0):
    
    deps, heads = get_dep_and_head(context)

    is_impossible = (start_pos == -1)
    query_tokens = tokenizer.tokenize(query)

    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    tok_to_orig_index = []
    orig_to_tok_index = []

    all_doc_tokens = []
    for (i, token) in enumerate(context.split(' ')):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)

        if len(all_doc_tokens) + len(sub_tokens) > max_tokens_for_doc:
            break

        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = -1
    tok_end_position = -1
    try:
        tok_start_position = orig_to_tok_index[start_pos]    # the answer is not in the current window
        tok_end_position = orig_to_tok_index[end_pos]
    except:
        tok_start_position = -1
        tok_end_position = -1

    tokens = []
    token_to_orig_map = {}
    segment_ids = []

    tokens.append(cls_token)
    segment_ids.append(cls_token_segment_id)
    cls_index = 0

    for token in query_tokens:
        tokens.append(token)
        segment_ids.append(sequence_a_segment_id)

    tokens.append(sep_token)
    segment_ids.append(sequence_a_segment_id)

    for i in range(len(all_doc_tokens)):
        token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
        tokens.append(all_doc_tokens[i])
        if tok_to_orig_index[i] >= trigger_s and tok_to_orig_index[i] <= trigger_e:
            segment_ids.append(sequence_a_segment_id)
        else:
            segment_ids.append(sequence_b_segment_id)

    # SEP token
    tokens.append(sep_token)
    segment_ids.append(sequence_b_segment_id)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(pad_token)
        input_mask.append(0)
        segment_ids.append(pad_token_segment_id)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    doc_offset = len(query_tokens) + 2

    if is_impossible:
        start_position = cls_index
        end_position = cls_index
    else:
        start_position = tok_start_position + doc_offset
        end_position = tok_end_position + doc_offset
    
    deps = _find_idx_from_dic(deps, token_to_orig_map)
    heads = _find_idx_from_dic(heads, token_to_orig_map)

    return input_ids, input_mask, segment_ids, doc_offset, token_to_orig_map, start_position, end_position, deps, heads


def _build_one_example(sentence, trigger_s, trigger_e, s, e, role_name, max_seq_length=512, pre_defined_query=None):
    if pre_defined_query:
        query = pre_defined_query
    else:
        query = 'What is the %s of %s?' % (role_name, ' '.join(sentence[trigger_s: trigger_e + 1]))
    context = ' '.join(sentence)
    return build_bert_example(query, context, trigger_s, trigger_e, s, e, max_seq_length)


def build_corpus_examples_train(corpus_data):
    results = []
    for elem in corpus_data:
        doc_id, sentences, trigger, entities, args = elem
        long_sentence = [item for sublist in sentences for item in sublist] # flat word list

        trigger_s, trigger_e, trigger_type = trigger
        schema_arg = config.event_schema[trigger_type]
        for role_name in schema_arg:
            found = False
            for arg in args:
                s, e, link = arg
                link = link[link.find('arg')+5:]
                if role_name == link:
                    found = True
                    temp = _build_one_example(long_sentence, trigger_s, trigger_e, s, e, role_name, max_seq_length=512)
                    results.append([doc_id, role_name] + list(temp))

            if not found:
                s, e = -1, -1
                temp = _build_one_example(long_sentence, trigger_s, trigger_e, s, e, role_name, max_seq_length=512)
                results.append([doc_id, role_name] + list(temp))
    return results


def build_corpus_examples_test(corpus_data):
    results = []
    for elem in corpus_data:
        doc_id, sentences, trigger, entities, args = elem
        long_sentence = [item for sublist in sentences for item in sublist] # flat word list

        trigger_s, trigger_e, trigger_type = trigger
        schema_arg = config.event_schema[trigger_type]
        for role_name in schema_arg:
            s, e = -1, -1
            temp = _build_one_example(long_sentence, trigger_s, trigger_e, s, e, role_name, max_seq_length=512)
            results.append([doc_id, role_name] + list(temp))
    return results



def back_translation(corpus_data):
    results = []
    for elem in corpus_data:
        doc_id, sentences, trigger, entities, args = elem
        long_sentence = [item for sublist in sentences for item in sublist] # flat word list

        trigger_s, trigger_e, trigger_type = trigger
        schema_arg = config.event_schema[trigger_type]
        for role_name in schema_arg:
            query = 'What is the %s in the %s event?' % (role_name, ' '.join(long_sentence[trigger_s: trigger_e + 1]))
            results.append([doc_id, role_name, query])
    return results


# event_schema = {}
# for elem in train_data + dev_data + test_data:
#     sentences, trigger, entities, args = elem
#     event_schema.setdefault(trigger[-1], set())
#     for arg in args:
#         _, _, a = arg
#         a = a[a.find('arg')+5:] # evt090arg01killer
#         event_schema[trigger[-1]].add(a)
# print(event_schema)

def get_bt_result():
    my_set = {}
    with open('back_translation_result.txt') as filein:
        for line in filein:
            try:
                doc_id, role, previous, _, bt_result = line.strip().split('\t')
            except:
                print(line)
                continue
            my_set[(doc_id, role)] = bt_result
    return my_set


def build_corpus_examples_test_back_translation(corpus_data):
    my_set = get_bt_result()
    results = []
    for elem in corpus_data:
        doc_id, sentences, trigger, entities, args = elem
        long_sentence = [item for sublist in sentences for item in sublist] # flat word list

        trigger_s, trigger_e, trigger_type = trigger
        schema_arg = config.event_schema[trigger_type]
        for role_name in schema_arg:
            s, e = -1, -1
            # print(doc_id, role_name)
            if (doc_id, role_name) in my_set:
                bt_query = my_set[(doc_id, role_name)]
                temp = _build_one_example(long_sentence, trigger_s, trigger_e, s, e, role_name, max_seq_length=512, pre_defined_query=bt_query)
                results.append([doc_id, role_name] + list(temp))
    return results


if __name__ == '__main__':
    train_data = read_corpus('data/train.jsonlines')
    dev_data = read_corpus('data/dev.jsonlines')
    test_data = read_corpus('data/test.jsonlines')

    role_count = {}
    n_total = 0
    for elem in train_data:
        temp = elem[-1]
        for _, _, role_name in temp:
            role_name = role_name[role_name.find('arg')+5:]
            role_count.setdefault(role_name, 0)
            role_count[role_name] += 1
            n_total += 1

    for key in role_count:
        role_count[key] = role_count[key] / n_total
    
    l = [[key, role_count[key]] for key in role_count]
    l = sorted(l, key=lambda x: -x[1])
    for x, y in l:
        print(x, y)


    # # back translation
    # results = back_translation(train_data + dev_data + test_data)
    # with open('data_query.pk', 'wb') as f:
    #     pickle.dump(results, f)


    # train_data = build_corpus_examples_train(train_data)
    # dev_data = build_corpus_examples_test(dev_data)
    # test_data = build_corpus_examples_test(test_data)

    # f = open('data.pk','wb')
    # data = [train_data, dev_data, test_data]
    # pickle.dump(data, f)



    
    test_data = build_corpus_examples_test_back_translation(test_data)
    f = open('data_test_bt.pk','wb')
    data = test_data
    pickle.dump(data, f)

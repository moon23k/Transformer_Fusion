import os, json
from datasets import load_dataset
from transformers import BertTokenizerFast



def process(orig_data, tokenizer, volumn=34000):
    volumn_cnt, processed = 0, []
    min_len, max_len, max_diff = 10, 300, 50
    
    
    for elem in orig_data:
        src, trg = elem['en'].lower(), elem['de'].lower()
        src_len, trg_len = len(src), len(trg)

        #define filtering conditions
        min_condition = (src_len >= min_len) & (trg_len >= min_len)
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        dif_condition = abs(src_len - trg_len) < max_diff

        if max_condition & min_condition & dif_condition:
            temp_dict = dict()
            
            src_tokenized = tokenizer(src)
            trg_tokenized = tokenizer(trg)

            temp_dict['input_ids'] = en_tokenized['input_ids']
            temp_dict['attention_mask'] = en_tokenized['attention_mask']
            temp_dict['labels'] = de_tokenized['input_ids']
            
            processed.append(temp_dict)
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == volumn:
                break

    return processed



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-4000], data_obj[-4000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')
    


def main():
    tokenizer = BertTokenizerFast.from_pretrained('prajjwal1/bert-small', model_max_length=512)
    orig = load_dataset('wmt14', 'de-en', split='train')['translation']
    processed = process(orig, tokenizer)
    save_data(processed)



if __name__ == '__main__':
    main()
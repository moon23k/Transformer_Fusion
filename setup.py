import os, json
from datasets import load_dataset




def process_data(orig_data, volumn=101100):
    min_len = 10 
    max_len = 300
    max_diff = 50

    volumn_cnt = 0
    processed = []
    
    for elem in orig_data:
        temp_dict = dict()
        src, trg = elem['en'].lower(), elem['de'].lower()
        src_len, trg_len = len(src), len(trg)

        #define filtering conditions
        min_condition = (src_len >= min_len) & (trg_len >= min_len)
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        dif_condition = abs(src_len - trg_len) < max_diff

        if max_condition & min_condition & dif_condition:
            processed.append({'src': src, 'trg': trg})
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == volumn:
                break

    return processed



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-1100], data_obj[-1100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')



def main():
    orig = load_dataset(
        'wmt14', 'de-en', split='train'
    )['translation']    

    processed = process_data(orig)
    
    save_data(processed)



if __name__ == '__main__':
    main()
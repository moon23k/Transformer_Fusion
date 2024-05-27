import os, re, json, yaml
from datasets import load_dataset




def process_translation_data(data_volumn):
    #load original dataset
    nmt_data = load_dataset('wmt14', 'de-en', split='train')['translation']
    
    min_len = 10 
    max_len = 300
    max_diff = 50
    volumn_cnt = 0

    processed = []
    
    for elem in nmt_data:
        temp_dict = dict()
        x, y = elem['en'].strip().lower(), elem['de'].strip().lower()
        x_len, y_len = len(x), len(y)

        #Filtering Conditions
        min_condition = (x_len >= min_len) & (y_len >= min_len)
        max_condition = (x_len <= max_len) & (y_len <= max_len)
        dif_condition = abs(x_len - y_len) < max_diff

        if max_condition & min_condition & dif_condition:
            processed.append({'x': x, 'y':y})
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == data_volumn:
                break

    return processed 





def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-5100], data_obj[-5100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')




def main():
    data_volumn = 55100
    processed = process_translation_data(data_volumn)
    save_data(processed)




if __name__ == '__main__':
    main()
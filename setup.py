import os, json
from datasets import load_dataset



def save_datasets(train, valid, test):
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}
    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)



def filter_dataset(data, min_len=10, max_len=300):
    filtered = []
    for elem in data:
        temp_dict = dict()
        src_len, trg_len = len(elem['en']), len(elem['de'])
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        min_condition = (src_len >= min_len) & (trg_len >= min_len)

        if max_condition & min_condition:
            temp_dict['src'] = elem['en']
            temp_dict['trg'] = elem['de']
            filtered.append(temp_dict)

    return filtered



def main(downsize=True, sort=True):
    #Download datasets
    train = load_dataset('wmt14', 'de-en', split='train')['translation']
    valid = load_dataset('wmt14', 'de-en', split='validation')['translation']
    test = load_dataset('wmt14', 'de-en', split='test')['translation']

    train = filter_dataset(train)
    valid = filter_dataset(valid)
    test = filter_dataset(test)

    if downsize:
        train = train[::100]

    if sort:
        train = sorted(train, key=lambda x: len(x['src']))
        valid = sorted(valid, key=lambda x: len(x['src']))
        test = sorted(test, key=lambda x: len(x['src']))

    save_datasets(train, valid, test)



if __name__ == '__main__':
    main()
    assert os.path.exists(f'data/train.json')
    assert os.path.exists(f'data/valid.json')
    assert os.path.exists(f'data/test.json')
import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence




class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)


    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        return self.data[idx]['src'], self.data[idx]['trg']




class Collator(object):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def tokenize(self, batch):
        return self.tokenizer(
            src_batch, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )


    def __call__(self, batch):
        src_batch, trg_batch = zip(*batch)

        src_encodings = self.tokenize(src_batch)
        trg_encodings = self.tokenize(trg_batch)

        return {'input_ids': src_encodings.input_ids, 
                'attention_mask': src_encodings.attention_mask,
                'labels': trg_encodings.input_ids}




def load_dataloader(config, tokenizer, split):

    is_train = split == 'train'
    batch_size = config.batch_size if is_train \
                 else config.batch_size // 4

    return DataLoader(
        Dataset(split), 
        batch_size=batch_size, 
        shuffle=True if is_train else False,
        collate_fn=Collator(config, tokenizer),
        pin_memory=True,
        num_workers=2
    )
import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, task, split):
        super().__init__()
        
        self.src, self.trg = task[:2], task[2:]
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = self.data[idx][f'{self.src}_ids']
        attention_mask = self.data[idx][f'{self.src}_mask']
        labels = self.data[idx][f'{self.trg}_ids']
        
        return input_ids, attention_mask, labels




class Collator(object):
    def __init__(self, config):
        self.task = config.task
        self.pad_id = config.pad_id

    def __call__(self, batch):
        ids_batch, masks_batch, labels_batch = [], [], []

        for ids, masks, labels in batch:
            ids_batch.append(torch.LongTensor(ids)) 
            masks_batch.append(torch.LongTensor(masks))
            labels_batch.append(torch.LongTensor(labels))

        ids_batch = pad_batch(ids_batch, pad_id)
        masks_batch = pad_batch(masks_batch, pad_id)
        labels_batch = pad_batch(labels_batch, pad_id)

        return {'input_ids': ids_batch, 
                'attention_mask': masks_batch,
                'labels': labels_batch}

    def pad_batch(batch):
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)


def load_dataloader(config, split):
    return DataLoader(Dataset(config.task, split), 
                      batch_size=config.batch_size, 
                      shuffle=True if config.mode=='train' else False,
                      collate_fn=collate_fn,
                      pin_memory=True,
                      num_workers=2)
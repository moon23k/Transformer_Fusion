import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split):
        super().__init__()
        self.task = config.task
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.task == 'ende':
            input_ids = self.data[idx]['en_ids']
            attention_mask = self.data[idx]['en_mask']
            labels = self.data[idx]['de_ids']
        else:
            input_ids = self.data[idx]['de_ids']
            attention_mask = self.data[idx]['de_mask']
            labels = self.data[idx]['en_ids']
        
        return input_ids, attention_mask, labels


def pad_batch(batch_list, pad_id):
    return pad_sequence(batch_list,
                        batch_first=True,
                        padding_value=pad_id)


def load_dataloader(config, split):
    global pad_id
    pad_id = config.pad_id    

    def collate_fn(batch):
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

    return DataLoader(Dataset(config, split), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=collate_fn,
                      num_workers=2,
                      pin_memory=True)
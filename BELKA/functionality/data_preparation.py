from torch.utils.data import DataLoader, Dataset
import torch


class BertDataset(Dataset):

    def __init__(self, embeddings, labels):

        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):

        return len(self.labels)
    
    def __getitem__(self, index):

        return {
            'input_ids': self.embeddings['input_ids'][index],
            'attention_mask': self.embeddings['attention_mask'][index],
            'label_ids': torch.tensor(self.labels[index], dtype=torch.float),
        }



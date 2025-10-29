from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import sys
from pathlib import Path
import torch

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils import preprocess

ADdatasetNamingDict = {
    '20NewsGroups' : ['SetFit/20_newsgroups'],
    'Reuters' : ['ucirvine/reuters21578', 'ModHayes'],
    'WOS' : ["HDLTex/web_of_science", 'WOS46985'],
    'DBpedia14' : ['fancyzhx/dbpedia_14'],
    'AGNews' : ['fancyzhx/ag_news']
}

class ADDataset(Dataset):
    def __init__(self, name, full_dataset=False, preprocessing=False):

        assert name in ('20NewsGroups', 'Reuters', 'WOS', 'DBpedia14', 'AGNews'), \
            f"dataset {name} doesn't exist, please verify the naming"
        
        self.name = name
        self.full_dataset = full_dataset
        self.preprocessing = preprocessing

        args_ = ADdatasetNamingDict[self.name]
        dataset = load_dataset(*args_)
        self.dataset = dataset

        if self.preprocessing:
            dataset = preprocess(dataset)
        
        if self.name == 'WOS':
            self.train_data = dataset['train']
            # just to had a generic 'text' column for all the datasets
            if 'input_data' in self.train_data.column_names:
                self.train_data = self.train_data.rename_column('input_data', 'text')
            self.test_data = None
        elif full_dataset:
            self.train_data = concatenate_datasets([self.train_data, self.test_data])
            self.test_data = None
        else:
            self.train_data = dataset['train']
            self.test_data = dataset['test']

        # just to had a generic 'text' column for all the datasets
        if self.name == 'DBpedia14':
            if 'content' in self.train_data.column_names:
                self.train_data = self.train_data.rename_column('content', 'text')
            if 'content' in self.test_data.column_names:
                self.test_data = self.test_data.rename_column('content', 'text')
        

    def get_splits(self):
        return self.train_data, self.test_data
    

EmbeddingMapping = {
    'bert' : 'bert_embedding',
    'glove' : 'glove_embedding',
    'fasttext' : 'fasttext_embedding',
    'tfidf' : 'tfidf_embedding'
}

class DatasetWrapper(Dataset):
    def __init__(self, dataset, type_emb):

        self.dataset = dataset
        self.texts = self.dataset['text']
        self.labels = self.dataset['anomaly_class']
        self.inputs = self.dataset[EmbeddingMapping[type_emb]]


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.inputs[idx]

        return torch.tensor(inputs), torch.tensor(label), text
    

class MergedDatasetWrapper(Dataset):
    def __init__(self, datasets):
        """
        datasets : liste de DatasetWrapper
        """
        self.texts = sum([ds.texts for ds in datasets], [])
        self.labels = torch.cat([torch.tensor(ds.labels) for ds in datasets])
        self.inputs = torch.cat([torch.tensor(ds.inputs) for ds in datasets])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx]), self.texts[idx]

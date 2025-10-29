from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
import re
import string
import unicodedata

def preprocess(dataset):

    def clean_text(text):

        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    splits = [split for split in ['train', 'test'] if split in dataset]

    for split in splits:
        if 'text' in dataset[split]:
            dataset[split]['text'] = dataset[split]['text'].apply(clean_text)

    return dataset


# Dataset Importing
#--------------------

def import_dataset(name="20NewsGroups", full_dataset_=False, batch_size=64):

    print(f"{name} dataset importing .... \n\n")

    # *****************************
    if name == "20NewsGroups":
        dataset = load_dataset("SetFit/20_newsgroups")

        # Nettoyage des textes
        dataset = dataset.map(lambda x: {"text": clean_corpus([x["text"]])[0] if clean_corpus([x["text"]]) else ""})
        dataset = dataset.filter(lambda x: len(x["text"]) > 0)

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)

        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
  
  # *****************************
    if name == "Reuters":

        dataset = load_dataset('ucirvine/reuters21578', 'ModHayes') 

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

  # *****************************
    if name == "WOS":

        dataset = load_dataset("HDLTex/web_of_science", 'WOS46985') 

        return DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)

  # *****************************
    if name == "DBpedia14":

        dataset = load_dataset("fancyzhx/dbpedia_14")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    # ***************************
    if name == "AGNews": 
        
        dataset = load_dataset("fancyzhx/ag_news")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
   


    raise Exception("The dataset naming doesn't correspond !")
    
    
    
# NLP Dataset Cleaning
#--------------------

def clean_corpus(
    corpus,
    lower=True,
    remove_punct=True,
    remove_digits=True,
):
    
    
    cleaned_corpus = []
    for doc in corpus:
        doc = unicodedata.normalize('NFKD', doc)
        doc = doc.encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        if lower:
            doc = doc.lower()
        if remove_punct:
            doc = doc.translate(str.maketrans('', '', string.punctuation))
        if remove_digits:
            doc = re.sub(r'\d+', '', doc)
        
        doc = re.sub(r'\s+', ' ', doc).strip()
        
        tokens = doc.split()
        
        cleaned_text = " ".join(tokens)
        
        # delete empty docs
        if cleaned_text.strip():
            cleaned_corpus.append(cleaned_text)
    
    return cleaned_corpus


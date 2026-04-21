from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
import re
import string
import unicodedata
from Data_Preparation.Tac.tac import textual_anomaly_contamination
from Data_Preparation.Embedding.embedding_encoder import EmbeddingEncoder
import numpy as np
from torch import Tensor
from Data_Preparation.Tac import tac
import torch
from tqdm import tqdm
import json
import os
from datasets import Dataset, Features, Value
from datasets import ClassLabel



# def preprocess(dataset):

#     def clean_text(text):

#         text = text.lower()
#         text = text.translate(str.maketrans("", "", string.punctuation))
#         text = re.sub(r'<.*?>', ' ', text)  
#         text = re.sub(r'\d+', ' ', text)  
#         text = re.sub(r'\W+', ' ', text) 
#         text = re.sub(r'\s+', ' ', text) 

#         return text.strip()

#     for split in ['train', 'test']:
#         if split in dataset and 'text' in dataset[split]:

#             dataset[split]['text'] = dataset[split]['text'].apply(clean_text)

#             dataset[split] = dataset[split][dataset[split]['text'] != ""]

#             dataset[split] = dataset[split].reset_index(drop=True)

#     return dataset

# from nltk.corpus import stopwords


def unify_text_column(dataset, dataset_name):

    if dataset_name == "sst2":
        return dataset.rename_column("sentence", "text")

    if dataset_name == "sms":
        return dataset.rename_column("sms", "text")

    return dataset

def preprocess(dataset, column='text'):

    def clean_text(example):
        text = example[column]

        if text is None:
            text = ""

        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\W+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        example[column] = text
        return example

    # Appliquer le nettoyage
    dataset = dataset.map(clean_text)

    # Supprimer les textes vides
    dataset = dataset.filter(lambda x: x[column] != "")

    return dataset


@torch.no_grad()
def encode_tokens(
    model,
    tokenizer,
    texts,
    device,
    batch_size=32,
    max_length=256,
    return_attentions=False,
    use_fp16=False,           # ✅ moitié mémoire
    save_to_disk=None,       # ✅ ex: "embeddings.pt" pour éviter OOM
):
    # Cast model en fp16 si demandé
    if use_fp16:
        model = model.half()

    all_embeddings = []
    all_tokens = []
    all_attention_masks = []
    all_attentions = [] if return_attentions else None

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs, output_attentions=return_attentions)

        last_hidden_state = outputs.last_hidden_state  # (B, T, D)

        # Normalisation + descente CPU immédiate
        emb = torch.nn.functional.normalize(last_hidden_state, p=2, dim=2)
        all_embeddings.append(emb.cpu())
        all_attention_masks.append(inputs["attention_mask"].cpu())

        tokens = [
            tokenizer.convert_ids_to_tokens(ids.tolist())
            for ids in inputs["input_ids"]
        ]
        all_tokens.extend(tokens)

        if return_attentions:
            batch_attn = torch.stack(outputs.attentions, dim=1)  # (B, L, H, T, T)
            batch_attn = batch_attn.mean(dim=(1, 2)).cpu()        # (B, T, T)
            all_attentions.append(batch_attn)

        # Libération propre — on libère les inputs aussi !
        del outputs, inputs
        # ❌ pas de empty_cache() ici

    # Concat en CPU — ✅ jamais remettre sur GPU
    embeddings = torch.cat(all_embeddings, dim=0)          # reste CPU
    attention_masks = torch.cat(all_attention_masks, dim=0)

    # Sauvegarde disque optionnelle pour les très gros datasets
    if save_to_disk:
        torch.save({
            "embeddings": embeddings,
            "tokens": all_tokens,
            "attention_masks": attention_masks,
        }, save_to_disk)
        print(f"✅ Sauvegardé dans {save_to_disk}")

    if return_attentions:
        attentions = torch.cat(all_attentions, dim=0)
        return embeddings, all_tokens, attentions, attention_masks
    else:
        return embeddings, all_tokens, attention_masks

# @torch.no_grad()
# def encode_tokens(
#     model,
#     tokenizer,
#     texts,
#     device,
#     batch_size=16,
#     max_length=128,
#     return_attentions=False  # 🔥 booléen ici
# ):
#     all_embeddings = []
#     all_tokens = []
#     all_attention_masks = []

#     # ⚠️ seulement si activé
#     all_attentions = [] if return_attentions else None

#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i+batch_size]

#         inputs = tokenizer(
#             batch,
#             padding="max_length",
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt"
#         ).to(device)

#         # 🔥 activer output_attentions seulement si nécessaire
#         outputs = model(
#             **inputs,
#             output_attentions=return_attentions
#         )

#         last_hidden_state = outputs.last_hidden_state  # (B, T, D)

#         emb = torch.nn.functional.normalize(last_hidden_state, p=2, dim=2)
#         all_embeddings.append(emb.cpu())

#         # ✅ Attention masks (toujours utiles)
#         all_attention_masks.append(inputs["attention_mask"].cpu())

#         # ✅ Tokens
#         tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in inputs["input_ids"]]
#         all_tokens.extend(tokens)

#         # 🔥 Calcul des attentions uniquement si demandé
#         if return_attentions:
#             batch_attn = torch.stack(outputs.attentions, dim=1)  # (B, L, H, T, T)
#             batch_attn = batch_attn.mean(dim=(1, 2)).cpu()       # (B, T, T)
#             all_attentions.append(batch_attn)

#         # 🔥 libération mémoire GPU (important)
#         del outputs
#         torch.cuda.empty_cache()

#     # 🔥 concat final
#     embeddings = torch.cat(all_embeddings, dim=0).to(device)
#     attention_masks = torch.cat(all_attention_masks, dim=0)

#     if return_attentions:
#         attentions = torch.cat(all_attentions, dim=0)
#         return embeddings, all_tokens, attentions, attention_masks
#     else:
#         return embeddings, all_tokens, attention_masks
        

# Dataset Importing
#--------------------

def import_dataset(name="20newsgroups", full_dataset_=False, batch_size=64):

    print(f"{name} dataset importing .... \n\n")

    # *****************************
    if name == "20newsgroups":
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
    if name == "reuters":

        dataset = load_dataset('ucirvine/reuters21578', 'ModApte', trust_remote_code=True)  #ModHayes  ModLewis

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

  # *****************************
    if name == "wos":

        dataset = load_dataset("HDLTex/web_of_science", 'WOS46985') 

        return DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)

  # *****************************
    if name == "dbpedia14":

        dataset = load_dataset("fancyzhx/dbpedia_14")
        # dataset = load_dataset("dbpedia_14")
        

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    # ***************************
    if name == "agnews": 
        
        dataset = load_dataset("fancyzhx/ag_news")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader


    # ***************************
    if name == "sms": 
        
        dataset = load_dataset("ucirvine/sms_spam")

        dataset_split = dataset['train'].train_test_split(
            test_size=0.2,              
            stratify_by_column="label", 
            seed=42                     
        )

        train_dataloader = DataLoader(dataset_split["train"], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset_split["test"], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    # ***************************
    if name == "enron": 
        
        dataset = load_dataset("SetFit/enron_spam")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    
    # ***************************
    if name == "imdb": 
        
        dataset = load_dataset("stanfordnlp/imdb")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    

    # ***************************
    if name == "sst2": 
        
        dataset = load_dataset("stanfordnlp/sst2")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        # because the testset is not labeled, the validation one is
        test_dataloader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['validation']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    
    # ***************************
    if name == "mage": 
        
        dataset = load_dataset("yaful/MAGE")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    

    # ***************************
    if name == "m4": 

        def load_jsonl(filepath):
            data = []
            with open(filepath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"[WARNING] skip ligne {i} dans {filepath}")
            return data


        def parse_filename(filename):
            name = filename.replace(".jsonl", "")
            parts = name.split("_")

            domain = parts[0].lower()
            generator = parts[1].lower()

            return domain, generator

        def convert_entry(entry, file_domain, generator):
            samples = []
            domain = file_domain

            # human
            if entry.get("human_text"):
                samples.append({
                    "text": entry["human_text"],
                    "label": 1,
                    "generator": "human",
                    "inlier_topic": domain,   
                    "prompt": entry.get("prompt", ""),
                    "source_id": entry.get("source_ID", "")
                })

            # machine
            if entry.get("machine_text"):
                samples.append({
                    "text": entry["machine_text"],
                    "label": 0,
                    "generator": generator,
                    "inlier_topic": domain,  
                    "prompt": entry.get("prompt", ""),
                    "source_id": entry.get("source_ID", "")
                })

            return samples

        features = Features({
            "text": Value("string"),
            "label": Value("int64"),
            "generator": Value("string"),
            "inlier_topic": Value("string"),
            "prompt": Value("string"),
            "source_id": Value("string"),
        })

        def load_m4_dataset(data_dir="Anomaly Detection Framework/m4_data"):
            all_samples = []

            for file in os.listdir(data_dir):
                if not file.endswith(".jsonl"):
                    continue

                filepath = os.path.join(data_dir, file)

                file_domain, generator = parse_filename(file)
                raw_data = load_jsonl(filepath)

                for entry in raw_data:
                    all_samples.extend(
                        convert_entry(entry, file_domain, generator)
                    )

            dataset = Dataset.from_list(all_samples, features=features)

            return dataset


        def add_stratify_key(example):
            example["stratify_key"] = f"{example['inlier_topic']}_{example['label']}"
            return example


        def create_dataloaders(dataset, batch_size=32, test_size=0.2, seed=42):

            dataset = dataset.map(add_stratify_key)

            unique_keys = list(set(dataset["stratify_key"]))
            dataset = dataset.cast_column(
                "stratify_key",
                ClassLabel(names=unique_keys)
            )

            split = dataset.train_test_split(
                test_size=test_size,
                stratify_by_column="stratify_key",
                seed=seed
            )

            train_loader = DataLoader(
                split["train"],
                batch_size=batch_size,
                shuffle=True
            )

            test_loader = DataLoader(
                split["test"],
                batch_size=batch_size,
                shuffle=False
            )

            return train_loader, test_loader


        data = load_m4_dataset('/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/m4_data')

        return create_dataloaders(
            data,
            batch_size=16
            )


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


    
def data_preparation(args, logger, embedding_encoding=False):
    
    # to avoid circular import between the two modules
    from Data_Preparation.Dataset.ADdatasets import ADDataset, DatasetWrapper

    logger.info("################################")
    logger.info("Loading Dataset...")
    logger.info("################################\n")

    dataset = ADDataset(args.dataset_name, args.full_dataset_, args.preprocessing)
    is_full = args.full_dataset_ or args.dataset_name == 'wos'

    if is_full:
        dataset_main, _ = dataset.get_splits()
        split_type = "complet"
    else:
        dataset_train, dataset_test = dataset.get_splits()
        split_type = "train_test"

    logger.info("################################")
    logger.info("Textual Anomaly Contamination...")
    logger.info("#################################\n")

    def contaminate(ds, is_train):
        return textual_anomaly_contamination(
            ds, args.dataset_name, args.inlier_topic, args.type_tac, args.anomaly_rate, is_trainset=is_train
        )

    if is_full:
        inlier, anomaly = contaminate(dataset_main, True)
    else:
        inlier_train, anomaly_train = contaminate(dataset_train, True)
        test = contaminate(dataset_test, False)

    # ================================
    # --- No Embedding Encoding ------
    # ================================
    if not embedding_encoding:
        if args.training_mode == "two_classes":
            if is_full:
                return {"complet":  concatenate_datasets([inlier, anomaly])}
            else:
                return {"train": concatenate_datasets([inlier_train, anomaly_train]),
                         "test": test}
        else:
            if is_full:
                return {"inlier": inlier, 
                        "anomaly": anomaly}
            else:
                return {"inlier_train": inlier_train,
                         "anomaly_train": anomaly_train,
                         "test": test}
            
    # ================================
    # --- Embedding Encoding ---------
    # ================================
    logger.info("################################")
    logger.info("Embedding Encodage...")
    logger.info("#################################\n")

    emb_encoder = EmbeddingEncoder(args.emb_model, args.type_emb)

    def encode(ds):
        return emb_encoder.forward(ds)

    if is_full:
        inlier_emb, anomaly_emb = map(encode, (inlier, anomaly))
    else:
        inlier_train_emb, anomaly_train_emb = map(encode, (inlier_train, anomaly_train))
        test_emb = encode(test)

    logger.info("################################")
    logger.info("Dataloader Creation...")
    logger.info("#################################\n")

    if args.training_mode == "two_classes":
        if is_full:
            return {"complet": DatasetWrapper(concatenate_datasets([inlier_emb, anomaly_emb]), args.type_emb)}
        else:
            return {"train": DatasetWrapper(concatenate_datasets([inlier_train_emb, anomaly_train_emb]), args.type_emb), 
                    "test": DatasetWrapper(test_emb, args.type_emb)}
    else:
        if is_full:
            return {"inlier": DatasetWrapper(inlier_emb, args.type_emb),
                     "anomaly": DatasetWrapper(anomaly_emb, args.type_emb)}
        else:
            return {"inlier_train": DatasetWrapper(inlier_train_emb, args.type_emb), 
                    "anomaly_train": DatasetWrapper(anomaly_train_emb, args.type_emb), 
                    "test": DatasetWrapper(test_emb, args.type_emb)}



def train_test_val_split(train, test, inlier_topic, dataset_name, type_tac, anomaly_rate, verbose=False):
    
    train_inlier, train_anomaly = tac.textual_anomaly_contamination(train, dataset_name, inlier_topic, type_tac, anomaly_rate, True)

    n_inliers_val = int(0.1 * len(train_inlier))
    inlier_indices = np.random.choice(len(train_inlier), n_inliers_val, replace=False)
    val_inlier_dataset = train_inlier.select(inlier_indices)

    train_inlier = train_inlier.select([i for i in range(len(train_inlier)) if i not in inlier_indices])
    
    n_anomalies_val = int(n_inliers_val / 0.9 * 0.1)
    anomaly_indices = np.random.choice(len(train_anomaly), n_anomalies_val, replace=False)
    val_anomaly_dataset = train_anomaly.select(anomaly_indices)

    val_ = concatenate_datasets([val_inlier_dataset, val_anomaly_dataset]).shuffle(seed=42)
    
    if verbose:
        print("TRAINSET")
        print(train_inlier)
        print(train_anomaly)
    
    if verbose:
        print("\nVALSET")
        print(val_)
        print()

    test_ = tac.textual_anomaly_contamination(test, dataset_name, inlier_topic, type_tac, anomaly_rate, False)
    # print(test_.filter(lambda x : x['anomaly_class']== 0))
    # print(test_.filter(lambda x : x['anomaly_class']== 1))
    if verbose:
        print("TESTSET")
        print(test_)

    return train_inlier, train_anomaly, val_, test_

def get_embeddings(sentencebertEncoder, train_reuters_, test_reuters_, inlier_topic, dataset_name, type_tac, anomaly_rate, device, hm='all', text_column='text'):

    train_inlier_reuters, train_anomaly_reuters, val_reuters, test_reuters = train_test_val_split(train_reuters_, test_reuters_, inlier_topic, dataset_name, type_tac, anomaly_rate, False)
    
    if hm == 'cvdd':
        return train_inlier_reuters, train_anomaly_reuters, val_reuters, test_reuters

    if hm == 'all':
        train_inlier_reuters = sentencebertEncoder.forward(train_inlier_reuters, text_column)
        test_reuters = sentencebertEncoder.forward(test_reuters, text_column)
        X_inlier = Tensor(train_inlier_reuters['sbert_embeddings']).to(device)
        X_test =  Tensor(test_reuters['sbert_embeddings']).to(device)
        y_test = np.array(test_reuters['anomaly_class'])

        return train_inlier_reuters, test_reuters, X_inlier, X_test, y_test
    
    elif hm == 'train':
        train_inlier_reuters = sentencebertEncoder.forward(train_inlier_reuters, text_column)
        X_inlier = Tensor(train_inlier_reuters['sbert_embeddings']).to(device)
        
        
        return train_inlier_reuters, _, X_inlier, _, _
    
    elif hm == 'test':
        test_reuters = sentencebertEncoder.forward(test_reuters, text_column)
        X_test =  Tensor(test_reuters['sbert_embeddings']).to(device)
        y_test = np.array(test_reuters['anomaly_class'])
        
        return _, test_reuters, _, X_test, y_test 

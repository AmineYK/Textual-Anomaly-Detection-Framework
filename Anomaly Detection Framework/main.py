from Data_Preparation.Dataset.ADdatasets import ADDataset, CVDDDatasetWrapper
from Data_Preparation.Tac.tac import textual_anomaly_contamination
from Data_Preparation.Embedding.embedding_encoder import EmbeddingEncoder
import argparse
import logging
from torch.utils.data import DataLoader
import time
from transformers import AutoTokenizer
from Modelisation.Baselines.OCSVM import ocsvm
import Modelisation.evaluation as ev
import Modelisation.Baselines.CVDD.networks.utils as utils
from Modelisation.Baselines.CVDD.networks import embedding_layer, cvdd_Net
import torch
import numpy as np
from datasets import concatenate_datasets
from Data_Preparation.utils import data_preparation


# --- Configuration du logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re
import os

def save_results(args, auc, ap, fpr95,
                 output_dir="/home/youcefk251/My Thesis/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results",
                 filename="results.txt",
                 overwrite=None):  # naive, smart, None

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    existing_content = ""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing_content = f.read()

    # On inclut maintenant ad_model dans le pattern
    pattern = (
        rf"Dataset:\s*{re.escape(args.dataset_name)}\s*"
        rf"Inlier class:\s*{re.escape(args.inlier_topic)}\s*"
        rf"Embedding type:\s*{re.escape(args.type_emb)}\s*"
        rf"AD model:\s*{re.escape(args.ad_model)}"
    )

    new_block = (
        "========================================\n"
        f"Dataset:        {args.dataset_name}\n"
        f"Inlier class:   {args.inlier_topic}\n"
        f"Embedding type: {args.type_emb}\n"
        f"AD model:       {args.ad_model}\n"
        "----------------------------------------\n"
        f"AUC:            {auc:.4f}\n"
        f"Avg Precision:  {ap:.4f}\n"
        f"FPR@95:         {fpr95:.4f}\n"
        "========================================\n\n"
    )

    match = re.search(pattern, existing_content)
    
    if match:
        old_block_pattern = r"========================================\n" + pattern + r".*?========================================\n\n"
        old_block = re.search(old_block_pattern, existing_content, flags=re.DOTALL).group(0)

        old_auc_match = re.search(r"AUC:\s*([\d.]+)", old_block)
        old_auc = float(old_auc_match.group(1)) if old_auc_match else -1

        do_replace = False
        if overwrite == "naive":
            do_replace = True
        elif overwrite == "smart" and auc > old_auc:
            do_replace = True
        elif overwrite is None:
            do_replace = False

        if do_replace:
            existing_content = re.sub(old_block_pattern, new_block, existing_content, flags=re.DOTALL)
            print(f"Résultats mis à jour pour ({args.dataset_name}, {args.inlier_topic}, {args.type_emb}, {args.ad_model}).")
        else:
            print(f"Résultats existants non modifiés pour ({args.dataset_name}, {args.inlier_topic}, {args.type_emb}, {args.ad_model}).")
            return
    else:
        existing_content += new_block
        print(f"Nouveaux résultats ajoutés pour ({args.dataset_name}, {args.inlier_topic}, {args.type_emb}, {args.ad_model}).")

    with open(filepath, "w") as f:
        f.write(existing_content)






def cvdd_model_pipeline(data_train, data_test, attention_size, n_attention_heads, embedding_type, seq_len, batch_size, shuffle, tokenizer=None, vocab=None):


    # ================================
    # ------------ BERT --------------
    # ================================
    if embedding_type == 'bert':
        if tokenizer is not None:
            cvdd_dataset_train = CVDDDatasetWrapper(data_train, embedding_type='bert', tokenizer=tokenizer, seq_len=seq_len)
            cvdd_dataset_test = CVDDDatasetWrapper(data_test, embedding_type='bert', tokenizer=tokenizer, seq_len=seq_len)
            pretrained_model = embedding_layer.EmbeddingFactory.create('bert', bert_name='distilbert-base-uncased', trainable=False)
        else:
            raise Exception(f"when 'embedding_type' = '{embedding_type}', the parameters 'bert_name' and 'tokenizer' is required")

    # ================================
    # ----------- GLOVE --------------
    # ================================
    elif embedding_type == 'glove': 
        if vocab is not None:
            cvdd_dataset_train = CVDDDatasetWrapper(data_train, embedding_type='glove', vocab=vocab, seq_len=seq_len)
            cvdd_dataset_test = CVDDDatasetWrapper(data_test, embedding_type='glove', vocab=vocab, seq_len=seq_len)
            pretrained_model = embedding_layer.EmbeddingFactory.create('glove',
                                    glove_path='./Modelisation/Baselines/CVDD/embedding_models/glove.6B.300d.txt',
                                    vocab=vocab,
                                    embedding_dim=300,
                                    trainable=True)
        else:
            raise Exception(f"when 'embedding_type' = '{embedding_type}', the parameter 'vocab' is required")
        
    # ================================
    # ----------- FASTTEXT -----------
    # ================================
    elif embedding_type == 'fasttext':
        if vocab is not None:
            cvdd_dataset_train = CVDDDatasetWrapper(data_train, embedding_type='fasttext', vocab=vocab, seq_len=seq_len)   
            cvdd_dataset_test = CVDDDatasetWrapper(data_test, embedding_type='fasttext', vocab=vocab, seq_len=seq_len)   
            pretrained_model = embedding_layer.EmbeddingFactory.create('fasttext',
                                    fasttext_path='./Modelisation/Baselines/CVDD/embedding_models/wiki-news-300d-1M.vec',
                                    vocab=vocab,
                                    embedding_dim=300,
                                    trainable=True)
        else:
            raise Exception(f"when 'embedding_type' = '{embedding_type}', the parameter 'vocab' is required")
        
    else: raise Exception(f" the 'embedding_type' {embedding_type} is not possible with CVDD, please choose ('bert','glove','fasttext')")
        

    dl_train = DataLoader(cvdd_dataset_train, batch_size=batch_size, shuffle=shuffle)
    dl_test = DataLoader(cvdd_dataset_test, batch_size=batch_size, shuffle=shuffle)
    
    model = cvdd_Net.CVDDNet(pretrained_model, attention_size, n_attention_heads)

    return model, dl_train, dl_test



def main(args):

    # some exceptions 
    if args.ad_model == 'ocsvm' and args.training_mode == 'two_classes':
        raise Exception(f"Warning ! the 'training_mode' : '{args.training_mode}' is not possible with '{args.ad_model}' model")
    if args.ad_model == 'cvdd' and args.training_mode == 'two_classes':
        raise Exception(f"Warning ! the 'training_mode' : '{args.training_mode}' is not possible with '{args.ad_model}' model")

    start = time.time()

    logger.info(
        f"\nStarting execution with dataset='{args.dataset_name}', "
        f"training_mode='{args.training_mode}', "
        f"inlier_topic='{args.inlier_topic}', "
        f"type_tac='{args.type_tac}', "
        f"anomaly_rate={args.anomaly_rate}, "
        f"embedding='{args.type_emb}' ({args.emb_model}). \n\n"
    )

    # if 'ad_model' is 'oscvm' : required_encoding is True else False 
    required_encoding = args.ad_model == 'ocsvm'
        
    dp_dict = data_preparation(args, logger, embedding_encoding=required_encoding)
    print(dp_dict, end="\n\n")

    # training_mode = 'one_class' --> return train/test in any dataset there is anomaly and inlier subset
    # training_mode = 'two_classes' --> return train/test and separate anomaly and inlier subset to get 4 dataloaders

    end = time.time()
    logger.info(f"Data Preparation ends after : {end - start:.2f} seconds")

    if args.training_mode == 'one_class':
        if args.full_dataset_ or args.dataset_name == 'WOS':
            dataset_inlier = dp_dict['inlier']
            dataset_anomaly = dp_dict['anomaly']

            data_train = dataset_inlier
        else:
            inlier_dataset_train = dp_dict['inlier_train']
            anomaly_dataset_train = dp_dict['anomaly_train']

            data_test = dp_dict['test']
            data_train = inlier_dataset_train

    if args.ad_model == 'ocsvm':

        ocsvm_kwargs = {
        "nu": args.nu,
        "kernel": args.kernel,
        "gamma": args.gamma
        }
        clf, y_pred_train, scores_train = ocsvm.One_Class_SVM(data_train.inputs, ocsvm_kwargs)

        y_pred_test = clf.predict(data_test.inputs)           
        scores_test = clf.decision_function(data_test.inputs)

        auc, ap, fpr95 = ev.evaluation(data_test.labels, scores_test, verbose=False)

        print(clf, end="\n\n")

        print(auc)
        print(ap)
        print(fpr95)

        save_results(args, auc, ap, fpr95,
                 output_dir="/home/youcefk251/My Thesis/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results",
                 filename="results.txt",
                 overwrite="smart")

    elif args.ad_model =='cvdd':

        # data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=args.shuffle)

        if args.type_emb == 'bert':
            tokenizer = AutoTokenizer.from_pretrained(args.emb_model)
            vocab = None

        elif args.type_emb in ('glove', 'fasttext'):
            corpus = data_train['text']
            vocab = utils.build_vocab(corpus,min_freq=1)
            tokenizer = None

        model, dl_train, dl_test = cvdd_model_pipeline(data_train, data_test, args.attention_size, args.n_attention_heads, args.type_emb, 200, args.batch_size, args.shuffle, tokenizer, vocab)

        cvdd_trainer = cvdd_Net.CVDDTrainer(optimizer_name='adam', learning_rate=1e-2, lr_milestones=(20,25), n_epochs=30, 
                 lambda_p=0.0, alpha_scheduler='linear', weight_decay=1e-6)
        
        model_trained = cvdd_trainer.train(model, dl_train)
        auc, ap, fpr95, _ = cvdd_trainer.test(model_trained, dl_test, ad_score='context_dist_mean')

        save_results(args, auc, ap, fpr95,
                 output_dir="/home/youcefk251/My Thesis/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results",
                 filename="results.txt",
                 overwrite="smart")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="20NewsGroups",
        help="Dataset naming (ex: '20newsgroups', 'reuters', etc.)"
    )

    parser.add_argument(
        "--training_mode",
        type=str,
        default="one_class",
        help="The training mode in order to welll seperate datasets"
    )

    parser.add_argument(
        "--full_dataset_",
        action="store_true",
        help="full dataset"
    )

    parser.add_argument(
        "--preprocessing",
        action="store_true",
        help="preprocessing function"
    )

    parser.add_argument(
        "--inlier_topic",
        type=str,
        default="science",
        help="The inlier category of the dataset"
    )

    parser.add_argument(
        "--type_tac",
        type=str,
        default="ruff",
        help="The type of anomaly contamintion for the dataset"
    )

    parser.add_argument(
        "--anomaly_rate",
        type=float,
        default=0.1,
        help="The rate of anomaly samples in the final dataset"
    )

    parser.add_argument(
        "--emb_model",
        type=str,
        default="distilbert-base-uncased",
        help="The name of the model"
    )

    parser.add_argument(
        "--type_emb",
        type=str,
        default="bert",
        help="The type of embedding encodage"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size"
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="suffle for dataloader"
    )

    parser.add_argument(
        "--ad_model",
        type=str,
        default="pass",
        help="The AD model"
    )

    args, remaining_argv = parser.parse_known_args()

    if args.ad_model == "cvdd":
        
        parser.add_argument("--attention_size", type=int, default=300, help="Attention dimension for CVDD model")
        parser.add_argument("--n_attention_heads", type=int, default=4, help="Number of attention heads")

    elif args.ad_model == "ocsvm":
        
        parser.add_argument("--nu", type=float, default=0.1, help="OCSVM nu parameter")
        parser.add_argument("--kernel", type=str, default="rbf", help="OCSVM kernel")
        parser.add_argument("--gamma", type=str, default="scale", help="OCSVM gamme parameter")




    args = parser.parse_args()
    main(args)

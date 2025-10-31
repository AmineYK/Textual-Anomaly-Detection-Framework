from Data_Preparation.Dataset.ADdatasets import ADDataset, CVDDDatasetWrapper, DatasetWrapper, MergedDatasetWrapper
from Data_Preparation.Tac.tac import textual_anomaly_contamination
from Data_Preparation.Embedding.embedding_encoder import EmbeddingEncoder
import argparse
import logging
from torch.utils.data import DataLoader, ConcatDataset
import time
from transformers import AutoTokenizer
from Modelisation.Baselines.OCSVM import ocsvm
import Modelisation.evaluation as ev
import Modelisation.Baselines.CVDD.networks.utils as utils
from Modelisation.Baselines.CVDD.networks import embedding_layer, cvdd_Net
import torch
import numpy as np


# --- Configuration du logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cvdd_model_pipeline(data_train, attention_size, n_attention_heads, embedding_type, seq_len, batch_size, shuffle, tokenizer=None, vocab=None):


    ################################
    ########### BERT ###############
    if embedding_type == 'bert':
        if tokenizer is not None:
            cvdd_dataset = CVDDDatasetWrapper(data_train, embedding_type='bert', tokenizer=tokenizer, seq_len=seq_len)
            pretrained_model = embedding_layer.EmbeddingFactory.create('bert', bert_name='distilbert-base-uncased', trainable=False)
        else:
            raise Exception(f"when 'embedding_type' = '{embedding_type}', the parameters 'bert_name' and 'tokenizer' is required")

    #################################
    ########### GLOVE ###############
    elif embedding_type == 'glove': 
        if vocab is not None:
            cvdd_dataset = CVDDDatasetWrapper(data_train, embedding_type='glove', vocab=vocab, seq_len=seq_len)
            pretrained_model = embedding_layer.EmbeddingFactory.create('glove',
                                    glove_path='./Modelisation/Baselines/CVDD/embedding_models/glove.6B.300d.txt',
                                    vocab=vocab,
                                    embedding_dim=300,
                                    trainable=False)
        else:
            raise Exception(f"when 'embedding_type' = '{embedding_type}', the parameter 'vocab' is required")
        
    ####################################
    ########### FASFTEXT ###############
    elif embedding_type == 'fasttext':
        if vocab is not None:
            cvdd_dataset = CVDDDatasetWrapper(data_train, embedding_type='fasttext', vocab=vocab, seq_len=seq_len)   
            pretrained_model = embedding_layer.EmbeddingFactory.create('fasttext',
                                    fasttext_path='./Modelisation/Baselines/CVDD/embedding_models/wiki-news-300d-1M.vec',
                                    vocab=vocab,
                                    embedding_dim=300,
                                    trainable=False)
        else:
            raise Exception(f"when 'embedding_type' = '{embedding_type}', the parameter 'vocab' is required")
        
    else: raise Exception(f" the 'embedding_type' {embedding_type} is not possible with CVDD, please choose ('bert','glove','fasttext')")
        

    dl = DataLoader(cvdd_dataset, batch_size=batch_size, shuffle=shuffle)
    
    model = cvdd_Net.CVDDNet(pretrained_model, attention_size, n_attention_heads)

    return model, dl

def data_preparation(args, logger, embedding_encoding = False):

    logger.info("################################")
    logger.info("Loading Dataset...")
    logger.info("################################\n")

    dataset = ADDataset(args.dataset_name, args.full_dataset_, args.preprocessing)

    if args.full_dataset_ or args.dataset_name == 'WOS':
        dataset_complet, _ = dataset.get_splits()
        dataset_train, dataset_test = None, None
    else:
        dataset_train, dataset_test = dataset.get_splits()
        dataset_complet = None

    logger.info("################################")
    logger.info("Textual Anomaly Contamination...")
    logger.info("#################################\n")

    if dataset_complet is None:
        inlier_dataset_train, anomaly_dataset_train = textual_anomaly_contamination(
            dataset_train, args.dataset_name, args.inlier_topic, args.type_tac, args.anomaly_rate
        )
        # inlier_dataloader_train = DataLoader(inlier_dataset_train, batch_size=args.batch_size, shuffle=args.shuffle)
        # anomaly_dataloader_train = DataLoader(anomaly_dataset_train, batch_size=args.batch_size, shuffle=args.shuffle)

        inlier_dataset_test, anomaly_dataset_test = textual_anomaly_contamination(
            dataset_test, args.dataset_name, args.inlier_topic, args.type_tac, args.anomaly_rate
        )
        # inlier_dataloader_test = DataLoader(inlier_dataset_test, batch_size=args.batch_size, shuffle=args.shuffle)
        # anomaly_dataloader_test = DataLoader(anomaly_dataset_test, batch_size=args.batch_size, shuffle=args.shuffle)

        # if false the return will be plain texts --> without encoding
        if not embedding_encoding:

            if args.training_mode == 'two_classes':
                return {
                        "train": ConcatDataset([inlier_dataset_train, anomaly_dataset_train]),
                        "test": ConcatDataset([inlier_dataset_test, anomaly_dataset_test])
                    }
            else:
                return {
                    "inlier_train": inlier_dataset_train,
                    "anomaly_train": anomaly_dataset_train,
                    "inlier_test": inlier_dataset_test,
                    "anomaly_test": anomaly_dataset_test
                }

    else:
        inlier_dataset_complet, anomaly_dataset_complet = textual_anomaly_contamination(
            dataset_complet, args.dataset_name, args.inlier_topic, args.type_tac, args.anomaly_rate
        )
        # inlier_dataloader_complet = DataLoader(inlier_dataset_complet, batch_size=args.batch_size, shuffle=args.shuffle)
        # anomaly_dataloader_complet = DataLoader(anomaly_dataset_complet, batch_size=args.batch_size, shuffle=args.shuffle)

        # if false the return will be plain texts --> without encoding
        if not embedding_encoding:
            if args.training_mode == 'two_classes':
                return {
                            "complet": ConcatDataset([inlier_dataset_complet, anomaly_dataset_complet])
                    }
            else:
                return {
                            "inlier": inlier_dataset_complet,
                            "anomaly": anomaly_dataset_complet
                    }


    # continue the process of tokenization and embedding of the texts to get dataloaders with embeddings
    if embedding_encoding:

        logger.info("################################")
        logger.info("Embedding Encodage...")
        logger.info("#################################\n")

        emb_encoder = EmbeddingEncoder(args.emb_model, args.type_emb)

        if dataset_complet is None:
            inlier_dataset_train_emb = emb_encoder.forward(inlier_dataset_train)
            anomaly_dataset_train_emb = emb_encoder.forward(anomaly_dataset_train)
            inlier_dataset_test_emb = emb_encoder.forward(inlier_dataset_test)
            anomaly_dataset_test_emb = emb_encoder.forward(anomaly_dataset_test)
        else:
            inlier_dataset_complet_emb = emb_encoder.forward(inlier_dataset_complet)
            anomaly_dataset_complet_emb = emb_encoder.forward(anomaly_dataset_complet)

        logger.info("################################")
        logger.info("Dataloader Creation...")
        logger.info("#################################\n")

        if dataset_complet is not None:
            wrapper_inlier_complet = DatasetWrapper(inlier_dataset_complet_emb, args.type_emb)
            wrapper_anomaly_complet = DatasetWrapper(anomaly_dataset_complet_emb, args.type_emb)

            inlier_dataloader = DataLoader(wrapper_inlier_complet, batch_size=args.batch_size, shuffle=args.shuffle)
            anomaly_dataloader = DataLoader(wrapper_anomaly_complet, batch_size=args.batch_size, shuffle=args.shuffle)

            if args.training_mode == 'two_classes':
                
                combined_dataset = MergedDatasetWrapper([wrapper_inlier_complet, wrapper_anomaly_complet])
                combined_dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
                return {"complet": combined_dataloader}

            return {"inlier": inlier_dataloader, "anomaly": anomaly_dataloader}

        else:
            wrapper_inlier_train = DatasetWrapper(inlier_dataset_train_emb, args.type_emb)
            wrapper_anomaly_train = DatasetWrapper(anomaly_dataset_train_emb, args.type_emb)
            wrapper_inlier_test = DatasetWrapper(inlier_dataset_test_emb, args.type_emb)
            wrapper_anomaly_test = DatasetWrapper(anomaly_dataset_test_emb, args.type_emb)

            inlier_dataloader_train = DataLoader(wrapper_inlier_train, batch_size=args.batch_size, shuffle=args.shuffle)
            anomaly_dataloader_train = DataLoader(wrapper_anomaly_train, batch_size=args.batch_size, shuffle=args.shuffle)
            inlier_dataloader_test = DataLoader(wrapper_inlier_test, batch_size=args.batch_size, shuffle=args.shuffle)
            anomaly_dataloader_test = DataLoader(wrapper_anomaly_test, batch_size=args.batch_size, shuffle=args.shuffle)

            if args.training_mode == 'two_classes':
                combined_train_dataset = MergedDatasetWrapper([wrapper_inlier_train, wrapper_anomaly_train])
                combined_test_dataset = MergedDatasetWrapper([wrapper_inlier_test, wrapper_anomaly_test])

                combined_dataloader_train = DataLoader(combined_train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
                combined_dataloader_test = DataLoader(combined_test_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

                return {
                    "train": combined_dataloader_train,
                    "test": combined_dataloader_test
                }

            return {
                "inlier_train": inlier_dataloader_train,
                "anomaly_train": anomaly_dataloader_train,
                "inlier_test": inlier_dataloader_test,
                "anomaly_test": anomaly_dataloader_test
            }
    



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
            dataloader_inlier = dp_dict['inlier']
            dataloader_anomaly = dp_dict['anomaly']

            data_train = dataloader_inlier
        else:
            inlier_dataloader_train = dp_dict['inlier_train']
            anomaly_dataloader_train = dp_dict['anomaly_train']

            inlier_dataloader_test = dp_dict['inlier_test']
            anomaly_dataloader_test = dp_dict['anomaly_test']

            data_train = inlier_dataloader_train


    if args.ad_model == 'ocsvm':

        ocsvm_kwargs = {
        "nu": args.nu,
        "kernel": args.kernel,
        "gamma": args.gamma
    }
        clf, y_pred_train, scores_train = ocsvm.One_Class_SVM(data_train.dataset.inputs, ocsvm_kwargs)

        ds = ConcatDataset([inlier_dataloader_test.dataset, anomaly_dataloader_test.dataset])
        inputs_test = [x for x, _, _ in ds]
        labels_test = [y.item() for _, y, _ in ds]

        y_pred_test = clf.predict(inputs_test)           
        scores_test = clf.decision_function(inputs_test)

        auc, f1, precision, recall, fpr95 = ev.evaluation(labels_test, scores_test, y_pred_test, verbose=False)

        print(clf, end="\n\n")

        print(auc)
        print(f1)
        print(precision)
        print(recall)
        print(fpr95)

    elif args.ad_model =='cvdd':

        if args.type_emb == 'bert':
            tokenizer = AutoTokenizer.from_pretrained(args.emb_model)
            vocab = None

        elif args.type_emb in ('glove', 'fasttext'):
            corpus = data_train['text']
            vocab = utils.build_vocab(corpus,min_freq=1)
            tokenizer = None

        model, dl = cvdd_model_pipeline(data_train, args.attention_size, args.n_attention_heads, args.type_emb, 200, args.batch_size, args.shuffle, tokenizer, vocab)
        for batch in dl:
            inputs, labels, texts = batch
            print(inputs, end="\n\n")
            print(labels, end="\n\n")
            print(texts, end="\n\n")

            
            # GloVe / FastText / BERT
            x = inputs.transpose(0, 1)  # shape (seq_len, batch_size)
            print(x.shape)



            cosine_dists, context_weights, A = model(x)
            print(cosine_dists.shape)
            print(context_weights.shape)
            print(A.shape)

            break


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
        default="ocsvm",
        help="The AD model"
    )

    args, remaining_argv = parser.parse_known_args()

    if args.ad_model == "cvdd":
        
        parser.add_argument("--attention_size", type=int, default=300, help="Attention dimension for CVDD model")
        parser.add_argument("--n_attention_heads", type=int, default=4, help="Number of attention heads")

    elif args.ad_model == "ocsvm":
        
        parser.add_argument("--nu", type=float, default=0.5, help="OCSVM nu parameter")
        parser.add_argument("--kernel", type=str, default="rbf", help="OCSVM kernel")
        parser.add_argument("--gamma", type=str, default="scale", help="OCSVM gamme parameter")




    args = parser.parse_args()
    main(args)

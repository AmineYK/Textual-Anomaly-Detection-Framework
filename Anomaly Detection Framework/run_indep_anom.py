import argparse
import logging
import torch
import torch.nn as nn
from torch import Tensor
import time
import tensorflow as tf
import torch.nn.functional as F
import datasets
from Data_Preparation import utils
from Data_Preparation.Embedding import embedding_encoder
from Modelisation.Baselines.OCSVM.ocsvm import OCSVM
from utils import load_data_inlier, load_data_test, load_hyperparams, get_data_fasttext
from Modelisation.Baselines.RSRAE.model import RSRAE
from Modelisation.Baselines.AE.autoencoder import AE
from Modelisation.Baselines.TCCM.model import TCCM
# from Modelisation.Baselines.CVDD.networks.cvdd_Net import CVDDModel
# from Modelisation.Baselines.CVDD.model_sbert import CVDDModel
from Modelisation.Baselines.CVDD.networks.model_bert import CVDDModel
from Modelisation.Baselines.FATE.fate import FATEModel
from Modelisation.Baselines.DATE.date import DATEModel
from Modelisation.FlowMatching.flow_matching import BasicFlowMatching
from Modelisation.FlowMatching.flow_matching_transformers import FlowDiT, FlowMatchingTransformers
from Modelisation.FlowMatching.flow_matching_transformers_token import FlowDiTToken,  FlowMatchingTransformersToken
from utils import save_results
import numpy as np
import os
import datasets
from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModel
from Data_Preparation.utils import encode_tokens
import Modelisation.evaluation as ev


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset_topics_dict= {
    '20newsgroups' : ['computer', 'recreation', 'science', 'miscellaneous', 'politics', 'religion'],
    'reuters' : ['earn', 'trade', 'acq', 'money-fx', 'crude', 'ship', 'interest'],
    # 'reuters' : ['trade', 'money-fx', 'crude', 'ship', 'interest'],
    'agnews' : ['World', 'Sports', 'Business', 'Sci-Tech'] ,
    'dbpedia14' : ["Company", "Educational Institution", "Artist", "Athlete", "Office Holder", 
                  "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal", "Plant", "Album", "Film", "Written Work"],
    'sms' : ['normal'],
    'enron': ['normal'],
    'imdb' : ['positive', 'negative'],
    'sst2': ['positive', 'negative'],
    'mage': ['normal'],
    'm4': ["wikipedia", "arxiv", "wikihow", "reddit", "peerread"]

}
COL = 'text'
# COL = 'content'

def main(args):

    if args.remove_stopwords:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        nltk.download('punkt')
        nltk.download('stopwords')

        stop_words = set(stopwords.words('english'))  # ou 'french'

        def remove_stopwords(text):
            tokens = word_tokenize(text)
            filtered = [word for word in tokens if word.lower() not in stop_words]
            return " ".join(filtered)

        def remove_stopwords_batch(batch):
            batch[COL] = [remove_stopwords(t) for t in batch[COL]]
            return batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create_tables()
    # print("Using device:", device)

    # if we'll run one inlier category or all categories for the dataset
    if args.runall:
        inlier_topics = dataset_topics_dict[args.dataset_name]
    else:
        inlier_topics = [args.inlier_topic]

    save_dir = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Data"
    file_path_hyp = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results/hyperparams.txt"
    ft_path = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Modelisation/Baselines/CVDD/embedding_models/wiki-news-300d-1M.vec"

    print(f"\n<<<<<<<<<<<<<< {args.dataset_name} >>>>>>>>>>>>>>>>>>>\n")

    # for every inlier category 
    for i, inlier_topic in enumerate(inlier_topics):

        print(f"------------------------ {inlier_topic}({i+1}/{len(inlier_topics)}) -----------------------------")

        if args.fm:
            list_auc_fm = []
            list_fpr_fm = []
            list_ap_fm = []
            list_time_fm = []
        
        if args.fm_trans:
            list_auc_fm_trans = []
            list_fpr_fm_trans = []
            list_ap_fm_trans = []
            list_time_fm_trans = []

            methods = ["sum", "mediane", "topk", "max", "attention_weighted", "weights"]

            metrics = {
                m: {"auc": [], "fpr": [], "ap": []}
                for m in methods
            }

        if args.ocsvm:
            list_auc_ocsvm = []
            list_fpr_ocsvm = []
            list_ap_ocsvm = []
            list_time_ocsvm = []

        if args.rsrae:
            list_auc_rsrae = []
            list_fpr_rsrae = []
            list_ap_rsrae = []
            list_time_rsrae = []

        if args.ae:
            list_auc_ae = []
            list_fpr_ae = []
            list_ap_ae = []
            list_time_ae = []

        if args.tccm:
            list_auc_tccm = []
            list_fpr_tccm = []
            list_ap_tccm = []
            list_time_tccm = []

        if args.cvdd:
            list_auc_cvdd = []
            list_fpr_cvdd = []
            list_ap_cvdd = []
            list_time_cvdd = []

        if args.date:
            list_auc_date = []
            list_fpr_date = []
            list_ap_date = []
            list_time_date = []

        if args.fate:
            list_auc_fate = []
            list_fpr_fate = []
            list_ap_fate = []
            list_time_fate = []

        # RSRAE --> X_train is infected with some anomalies
        # if args.rsrae:
        #     X_inlier_anoma = load_data_inlier(args.dataset_name, inlier_topic, save_dir, True)        
        # if args.cvdd:
        #     data_train = load_data_inlier(args.dataset_name, inlier_topic, save_dir, is_infec=False, is_cvdd=True)
        # load the X_inlier matrix
        # X_inlier = load_data_inlier(args.dataset_name, inlier_topic, save_dir)





        data_train = load_data_inlier(args.dataset_name, inlier_topic, save_dir, is_infec=False, is_cvdd=True)

        if args.remove_stopwords:
            data_train = data_train.map(remove_stopwords_batch, batched=True)


        if args.fate:
            path = os.path.join(save_dir, f"{args.dataset_name}/{inlier_topic}/ds_train_{inlier_topic}_anomaly.pt")
            data_train_anomaly_fate = datasets.load_from_disk(path)

        if args.type_emb == "fasttext":
            X_inlier = get_data_fasttext(data_train, ft_path, device)
        
        elif args.type_emb == 'sentence-bert':
            X_inlier = Tensor(data_train['sbert_embeddings']).to(device)

        elif args.type_emb == 'bert':
            model_name = "roberta-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            bertmodel = AutoModel.from_pretrained(model_name).to(device)
            bertmodel.eval()

            # X_inlier, tokens_train, attentions_train_mask = encode_tokens(bertmodel, tokenizer, data_train[:10000][COL], device, 64, 256)
            X_inlier, tokens_train, attentions_train_mask = encode_tokens(bertmodel, tokenizer, data_train[COL], device, 64, 128)
            # X_inlier = X_inlier.mean(dim=1)
        else: raise Exception("type_emb must be completed !")

        if args.nu > 0:
            path = os.path.join(save_dir, f"{args.dataset_name}/{inlier_topic}/ds_train_{inlier_topic}_anomaly_{int(args.nu*100)}.pt")
            data_train_anomaly = datasets.load_from_disk(path)
            X_anom_for_train = Tensor(data_train_anomaly['sbert_embeddings']).to(device) 
            print(X_anom_for_train.shape)

            # bert
            # X_inlier_anomaly, _, _ = encode_tokens(bertmodel, tokenizer, data_train_anomaly[COL], device, 64, 256)
            # print("Anom : ")
            # print(X_inlier_anomaly.shape)


        print(X_inlier.shape)

        # print(f" Embedding : {args.type_emb} --> {X_inlier.shape}")

        # get the hyperparamter for the FM model for this specific inlier category
        hyp = load_hyperparams(args.dataset_name, inlier_topic, 'sentence_bert', file_path_hyp)
        # print(hyp)
        nb_runs = args.nb_runs
        for n_run in range(1, nb_runs):

            print(f"+++++++++++++++++++++ run : {n_run} / {nb_runs - 1} +++++++++++++++++\n")

            # if args.cvdd:
            #     data_test = load_data_test(args.dataset_name, inlier_topic, n_run, save_dir, is_cvdd=True)
            # X_test, y_test = load_data_test(args.dataset_name, inlier_topic, n_run, save_dir)

            data_test = load_data_test(args.dataset_name, inlier_topic, n_run, save_dir, is_cvdd=True)
            
            if args.remove_stopwords:
                data_test = data_test.map(remove_stopwords_batch, batched=True)

            # y_test = np.array(data_test['anomaly_class'][:12000])
            y_test = np.array(data_test['anomaly_class'])

            if args.type_emb == "fasttext":
                X_test = get_data_fasttext(data_test, ft_path, device)
        
            elif args.type_emb == 'sentence-bert':
                X_test = Tensor(data_test['sbert_embeddings']).to(device)
            
            elif args.type_emb == 'bert':
                # X_test,  tokens_test, attentions_test_mask  = encode_tokens(bertmodel, tokenizer, data_test[:5000][COL], device, 64, 256)
                X_test,  tokens_test, attentions_test_mask  = encode_tokens(bertmodel, tokenizer, data_test[COL], device, 64, 128)
                # X_test = X_test.mean(dim=1)

            else: raise Exception("type_emb must be completed !")
            
            print(X_test.shape)
            

            #########################################
            ################# OCSVM #################
            #########################################  

            if args.ocsvm:
                ocsvm_args = {
                    "nu": 0.1,
                    "kernel": 'rbf',
                    "gamma": 'scale'
                }

                ocsvm_model = OCSVM(ocsvm_args)
                
                taac = time.time()
                if args.nu > 0:   
                    _ = ocsvm_model.train(torch.concatenate([X_inlier, X_anom_for_train]).cpu())
                else:
                    # _ = ocsvm_model.train(X_inlier.cpu())
                    _ = ocsvm_model.train(X_inlier.mean(dim=1).cpu())
                tiic = time.time()
                print(f"\nOCSVM finishing... after {(tiic-taac)/60:.3f} mn")

                # auc_ocsvm, fpr95_ocsvm, ap_ocsvm = ocsvm_model.test(X_test.cpu(), y_test)
                auc_ocsvm, fpr95_ocsvm, ap_ocsvm = ocsvm_model.test(X_test.mean(dim=1).cpu(), y_test)
                print(f"OCSVM --> AUC: {auc_ocsvm:.4f} | FPR@95: {fpr95_ocsvm:.4f} | AP: {ap_ocsvm:.4f}\n")

                list_auc_ocsvm.append(auc_ocsvm)
                list_fpr_ocsvm.append(fpr95_ocsvm)    
                list_ap_ocsvm.append(ap_ocsvm)
                list_time_ocsvm.append((tiic-taac))


            # #########################################
            # ################# RSRAE #################
            # #########################################  

            if args.rsrae:

                rsrae_args = {
                    # "input_dim": X_inlier.shape[2], "hidden_layer_sizes": (64,32,16), "intrinsic_size": 10,
                    "input_dim": X_inlier.shape[2], "hidden_layer_sizes": (128,64,32), "intrinsic_size": 10,
                    "activation": nn.ReLU(), "norm_type": 'l21', "loss_norm_type": 'mse',
                    "if_rsr": True, "enforce_proj": True, "all_alt": True,
                    "learning_rate": 1e-4, "lambda1": 0.1, "lambda2": 0.1,
                    "epoch_size": 100, "batch_show": 50, "normalize": True,
                    "bn": False, "seed": 42, 'batch_size': X_inlier.shape[0] // 100
                }

                rsrae_model = RSRAE(rsrae_args)

                taac = time.time()
                if args.nu > 0.0:
                    # _ = rsrae_model.train(torch.concatenate([X_inlier, X_anom_for_train]), device)
                    _ = rsrae_model.train(torch.concatenate([X_inlier.mean(dim=1), X_inlier_anomaly.mean(dim=1)]), device)
                else:
                    # _ = rsrae_model.train(X_inlier, device)
                    _ = rsrae_model.train(X_inlier.mean(dim=1), device)
                    


                tiic = time.time()
                
                print(f"\nRSRAE finishing... after {(tiic-taac)/60:.3f} mn")

                # auc_rsrae, fpr95_rsrae, ap_rsrae = rsrae_model.test(X_test, y_test)
                auc_rsrae, fpr95_rsrae, ap_rsrae = rsrae_model.test(X_test.mean(dim=1), y_test, device)
                print(f"RSRAE --> AUC: {auc_rsrae:.4f} | FPR@95: {fpr95_rsrae:.4f} | AP: {ap_rsrae:.4f}\n")

                list_auc_rsrae.append(auc_rsrae)
                list_fpr_rsrae.append(fpr95_rsrae)    
                list_ap_rsrae.append(ap_rsrae) 
                list_time_rsrae.append((tiic-taac))


           # ######################################
           # ################# AE #################
           # ######################################

            if args.ae:

                ae_args = {
                    "contamination": 0.1,
                    "hidden_neuron_list": [64, 32, 16],
                    "hidden_activation_name": "relu",
                    "epoch_num": 30,
                    "batch_size": X_inlier.shape[0] // 5,
                    "dropout_rate": 0.0,
                    "verbose": 0
                }


                ae_model = AE(ae_args)

                taac = time.time()
                if args.nu > 0:   
                    _ = ae_model.train(torch.concatenate([X_inlier, X_anom_for_train]).cpu())
                else:
                    # _ = ae_model.train(X_inlier)
                    _ = ae_model.train(X_inlier.mean(dim=1))
                tiic = time.time()
                print(f"\nAE finishing... after {(tiic-taac)/60:.3f} mn")
                
                # auc_ae, fpr95_ae, ap_ae = ae_model.test(X_test, y_test)
                auc_ae, fpr95_ae, ap_ae = ae_model.test(X_test.mean(dim=1), y_test)
                print(f"AE --> AUC: {auc_ae:.4f} | FPR@95: {fpr95_ae:.4f} | AP: {ap_ae:.4f}\n")

                list_auc_ae.append(auc_ae)
                list_fpr_ae.append(fpr95_ae)    
                list_ap_ae.append(ap_ae)  
                list_time_ae.append((tiic-taac))


            ########################################
            ################# TCCM #################
            ########################################  

            if args.tccm:

                tccm_args={
                    # "n_features": X_inlier.shape[1],
                    "n_features": X_inlier.shape[2],
                    "epochs" : 50,
                    "learning_rate" : 1e-3,
                    "batch_size": 256,
                    "device": device
                }

                tccm_model = TCCM(tccm_args)

                taac = time.time()
                if args.nu > 0:   
                    _ = tccm_model.train(torch.concatenate([X_inlier, X_anom_for_train]))
                else:
                    # _ = tccm_model.train(X_inlier)
                    _ = tccm_model.train(X_inlier.mean(dim=1))
                tiic = time.time()
                print(f"\nTCCM finishing... after {(tiic-taac)/60:.3f} mn")
                
                # auc_tccm, fpr95_tccm, ap_tccm = tccm_model.test(X_test, y_test)
                auc_tccm, fpr95_tccm, ap_tccm = tccm_model.test(X_test.mean(dim=1), y_test)
                print(f"TCCM --> AUC: {auc_tccm:.4f} | FPR@95: {fpr95_tccm:.4f} | AP: {ap_tccm:.4f}\n")
                
                list_auc_tccm.append(auc_tccm)
                list_fpr_tccm.append(fpr95_tccm)    
                list_ap_tccm.append(ap_tccm)  
                list_time_tccm.append((tiic-taac))  



            ########################################
            ################# CVDD #################
            ########################################  

            # if args.cvdd:
            #     cvdd_args = {
            #         "type_emb": "fasttext",
            #         # "emb_model": "distilbert-base-uncased",
            #         "emb_model": "distilroberta-base",
            #         "attention_size": 150,
            #         "n_attention_heads": 10,
            #         "lr": 0.001,
            #         "weight_decay": 0,
            #         "lr_milestones": (15, 25),
            #         "n_epochs": 30,
            #         "lambda_p": 1.0,
            #         "alpha_scheduler": "logarithmic",
            #         "seq_len": 100,
            #         "batch_size": 64,
            #         "min_freq": 1,
            #         "device": device
            #     }

            #     cvdd_model = CVDDModel(cvdd_args)

            #     taac = time.time()
            #     cvdd_model_trained, cvdd_trainer = cvdd_model.train(data_train, data_test)
            #     tiic = time.time()
            #     print(f"\CVDD finishing... after {(tiic-taac)/60:.3f} mn")
                
            #     auc_cvdd, fpr95_cvdd, ap_cvdd = cvdd_model.test(cvdd_model_trained, cvdd_trainer, data_train, data_test)
            #     print(f"CVDD --> AUC: {auc_cvdd:.4f} | FPR@95: {fpr95_cvdd:.4f} | AP: {ap_cvdd:.4f}\n")


            # if args.cvdd:
            #     cvdd_args = {
            #     "n_attention_heads": 8,
            #     "latent_dim": 150,
            #     "lr": 1e-2,
            #     "n_epochs": 100,
            #     "lambda_p": 0.1,
            #     "batch_size": 64,
            #     "device": device
            #     }

            #     cvdd_model = CVDDModel(cvdd_args)

            #     taac = time.time()
            #     cvdd_model_trained = cvdd_model.train(data_train)
            #     tiic = time.time()
            #     print(f"\CVDD finishing... after {(tiic-taac)/60:.3f} mn")
                
            #     auc_cvdd, fpr95_cvdd, ap_cvdd = cvdd_model.test(data_test)
            #     print(f"CVDD --> AUC: {auc_cvdd:.4f} | FPR@95: {fpr95_cvdd:.4f} | AP: {ap_cvdd:.4f}\n")


            if args.cvdd:
                cvdd_args = {
                    "bert_name": "roberta-base", #albert-large-v2   
                    "hidden_size": 768, #1024 
                    "n_attention_heads": 10,
                    "attention_size": 64,
                    "freeze_bert": True,
                    "lr": 1e-3,
                    "weight_decay" : 0,
                    "lambda_p": 0.1,
                    "n_epochs": 30,
                    "batch_size": 64,
                    "device": device
                    }
                
                cvdd_model = CVDDModel(cvdd_args)

                taac = time.time()
                if args.nu > 0:
                    cvdd_model.train(concatenate_datasets([data_train, data_train_anomaly]))
                else:
                    cvdd_model.train(data_train, COL)
                tiic = time.time()
                print(f"\CVDD finishing... after {(tiic-taac)/60:.3f} mn")
                
                auc_cvdd, fpr95_cvdd, ap_cvdd = cvdd_model.test(data_test, COL)
                print(f"CVDD --> AUC: {auc_cvdd:.4f} | FPR@95: {fpr95_cvdd:.4f} | AP: {ap_cvdd:.4f}\n")
            
                list_auc_cvdd.append(auc_cvdd)
                list_fpr_cvdd.append(fpr95_cvdd)    
                list_ap_cvdd.append(ap_cvdd)  
                list_time_cvdd.append((tiic-taac))  



            ########################################
            ################# DATE #################
            ########################################  

            if args.date:
                date_args = {
                    # "which_config": "bert",
                    # "encoder_name": "albert-base-v2", 
                    "which_config": "roberta",
                    "encoder_name":  "roberta-base",
                    # "which_config": "electra",
                    # "encoder_name": "google/electra-small-discriminator",
                    "K": 20,
                    "lr": 1e-3,
                    "weight_decay" : 1e-4,
                    "seq_len": 256,
                    "ratio": 0.50,
                    "n_epochs": 50,
                    "batch_size": 64,
                    "device": device
                    }
                            
                date_model = DATEModel(date_args)

                taac = time.time()
                if args.nu > 0:
                    date_model.train(concatenate_datasets([data_train, data_train_anomaly]))
                else:
                    date_model.train(data_train, COL)
                tiic = time.time()
                print(f"\DATE finishing... after {(tiic-taac)/60:.3f} mn")
                
                auc_date, fpr95_date, ap_date = date_model.test(data_test, COL)
                print(f"DATE --> AUC: {auc_date:.4f} | FPR@95: {fpr95_date:.4f} | AP: {ap_date:.4f}\n")
            
                list_auc_date.append(auc_date)
                list_fpr_date.append(fpr95_date)    
                list_ap_date.append(ap_date)  
                list_time_date.append((tiic-taac))  

            ########################################
            ################# FATE #################
            ########################################  

            if args.fate:
                data_test_inlier = data_test.filter(lambda x: x["anomaly_class"] == 0)
                data_test_anomaly = data_test.filter(lambda x: x["anomaly_class"] == 1)

                if args.nu > 0:
                    data_train_ = concatenate_datasets([data_train, data_train_anomaly])
                else:
                    data_train_ = data_train
                fate_args = {
                    "device": device,
                    "batch_size": 128,
                    "n_epochs": 15,
                    "lr": 1e-3,
                    "include_regularization": True,
                    "top_k": 0.1,
                    "nb_shot": 10,
                    "train_inlier_text": data_train_[COL],     
                    "train_anomaly_text": data_train_anomaly_fate[COL],
                    "test_inlier_text": data_test_inlier[COL],
                    "test_anomaly_text": data_test_anomaly[COL]
                }
                            
                fate_model = FATEModel(fate_args)

                taac = time.time()
                fate_model.train()
                tiic = time.time()
                print(f"\FATE finishing... after {(tiic-taac)/60:.3f} mn")
                
                auc_fate, fpr95_fate, ap_fate = fate_model.test()
                print(f"FATE --> AUC: {auc_fate:.4f} | FPR@95: {fpr95_fate:.4f} | AP: {ap_fate:.4f}\n")
            
                list_auc_fate.append(auc_fate)
                list_fpr_fate.append(fpr95_fate)    
                list_ap_fate.append(ap_fate)  
                list_time_fate.append((tiic-taac))  


            #################################################
#          ################# Flow Matching #################
#          ################################################# 

            if args.fm:
                fm_args = {
                    "batch_size": hyp["batch_size"],
                    "input_dim": X_inlier.shape[1],
                    "latent_dim": hyp["latent_dim"],
                    "sinu": False,
                    "batchnorm": False,
                    "dropout": hyp["dropout"],
                    "lr": hyp["lr"],
                    "weight_decay": hyp["weight_decay"],
                    "n_epochs": hyp["n_epochs"],
                    "target": X_inlier.cpu(),
                    "source": hyp["source"],
                    "device": device,
                    "score_type" : "norm",
                    "solver_type" : "midpoint",
                    "n_steps": 10, 
                    "noise_is_target": False
                }

                fm_model = BasicFlowMatching(fm_args)

                taac = time.time()
                _ = fm_model.train(X_inlier)
                tiic = time.time()
                print(f"\nFM finishing... after {(tiic-taac)/60:.3f} mn")

                auc_fm, fpr95_fm, ap_fm = fm_model.test(X_test, y_test)
                print(f"FM --> AUC: {auc_fm:.4f} | FPR@95: {fpr95_fm:.4f} | AP: {ap_fm:.4f}\n")
        
                list_auc_fm.append(auc_fm)
                list_fpr_fm.append(fpr95_fm)    
                list_ap_fm.append(ap_fm)
                list_time_fm.append((tiic-taac))


            ######################################################
#          ################ Flow Matching Transformers ###########
#          ####################################################### 
            if args.fm_trans:
                config = {
                    'latent_dim': 768,
                    'hidden_dim': 64,
                    'depth': 4,
                    'n_heads': 4,
                    'lr': 1e-3,
                    'weight_decay': 1e-3,
                    'lambda_svdd': 1e-2,
                    # 'lambda_push': 1e-2,
                    'lambda_push': 0,
                    'lambda_margin': 0,
                    'epochs': 300,
                    'lr_epochs':100,
                    'warmup_epochs': -1,
                    'grad_clip': 1.0,
                    'flow_type': 'linear',  
                    'sigma': 0.1, 
                    'batch_size' : 32,
                    'lambda_reg_angle': None,
                    'lambda_reg_kl': None,
                    'n_step_euler_integrate':1,
                    'coef_var': 1,
                    'rate_neg_batch':1.0,
                    'sig_levels_neg' : [0.5, 0.7],
                    'target' : 'gaussian-neigh',
                    'source' : X_inlier.to(device)
                }

                model = FlowDiTToken(
                    latent_dim=config['latent_dim'],
                    hidden_dim=config['hidden_dim'],
                    depth=config['depth'],
                    n_heads=config['n_heads']
                ).to(device)

                fm_transformer = FlowMatchingTransformersToken(model, config['source'], config['target'], config, noise_is_target=True, rectified=None)

                taac = time.time()
                fm_transformer.train(attentions_train_mask.to(device), True)
                tiic = time.time()
                print(f"\nFM Transformer finishing... after {(tiic-taac)/60:.3f} mn")


                # for method in methods:                    
                #     if method == "topk":
                #         auc, fpr, ap = fm_transformer.test(
                #             X_test,
                #             attentions_test_mask.to(device),
                #             y_test,
                #             method,
                #             k_rate=0.3
                #         )
                        
                #     elif method in ["attention_weighted", "weights"]:
                #         auc, fpr, ap = fm_transformer.test(
                #             X_test,
                #             attentions_test_mask.to(device),
                #             y_test,
                #             method,
                #             None,
                #             attentions_test.to(device)
                #         )
                        
                #     else:
                #         auc, fpr, ap = fm_transformer.test(
                #             X_test,
                #             attentions_test_mask.to(device),
                #             y_test,
                #             method
                #         )

                #     metrics[method]["auc"].append(auc)
                #     metrics[method]["fpr"].append(fpr)
                    # metrics[method]["ap"].append(ap)


                # auc_fm_trans, fpr95_fm_trans, ap_fm_trans = fm_transformer.test(X_test, y_test, X_inlier, 'norm-centroid')
                x_final, _, _ = fm_transformer.euler_integrate(X_test.to(device), attentions_test_mask.to(device), 15, False)

                x_1_test = x_final.cpu().numpy()
                scores = np.sum((x_1_test - fm_transformer.centroid.repeat(x_1_test.shape[0],1).cpu().numpy()) ** 2, axis=1)
                auc_fm_trans, fpr95_fm_trans, ap_fm_trans = ev.evaluation(y_test, scores)

                print(f"FM --> AUC: {auc_fm_trans:.4f} | FPR@95: {fpr95_fm_trans:.4f} | AP: {ap_fm_trans:.4f}\n")

                list_auc_fm_trans.append(auc_fm_trans)
                list_fpr_fm_trans.append(fpr95_fm_trans)    
                list_ap_fm_trans.append(ap_fm_trans)
                list_time_fm_trans.append((tiic-taac))

        if args.fm:
            print(inlier_topic, np.mean(list_auc_fm))
            save_results(
                dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb=args.type_emb ,ad_model="flow-matching",
                auc_mean=np.mean(list_auc_fm), ap_mean=np.mean(list_ap_fm),fpr_mean=np.mean(list_fpr_fm),
                auc_std = np.std(list_auc_fm),ap_std =  np.std(list_ap_fm),fpr_std = np.std(list_fpr_fm),
                train_time = np.mean(list_time_fm),nu=args.nu,overwrite='smart'
                )
            
        if args.fm_trans:
            print(inlier_topic, np.mean(list_auc_fm_trans), np.mean(list_fpr_fm_trans), np.mean(list_ap_fm_trans))
            save_results(
                dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb=args.type_emb ,ad_model="flow-matching-Transformers-Comp",
                auc_mean=np.mean(list_auc_fm_trans), ap_mean=np.mean(list_ap_fm_trans),fpr_mean=np.mean(list_fpr_fm_trans),
                auc_std = np.std(list_auc_fm_trans),ap_std =  np.std(list_ap_fm_trans),fpr_std = np.std(list_fpr_fm_trans),
                train_time = np.mean(list_time_fm_trans), nu=args.nu, overwrite='smart'
                )
            # for method in methods:
            #     print(np.mean(metrics[method]["auc"]), np.mean(metrics[method]["ap"]), np.mean(metrics[method]["fpr"]))
            #     save_results(
            #         dataset_name=args.dataset_name,
            #         inlier_topic=inlier_topic,
            #         type_emb=args.type_emb,
            #         ad_model=f"FMTToken-{method}",
                    
            #         auc_mean=np.mean(metrics[method]["auc"]),
            #         ap_mean=np.mean(metrics[method]["ap"]),
            #         fpr_mean=np.mean(metrics[method]["fpr"]),
                    
            #         auc_std=np.std(metrics[method]["auc"]),
            #         ap_std=np.std(metrics[method]["ap"]),
            #         fpr_std=np.std(metrics[method]["fpr"]),
                    
            #         train_time=np.mean(list_time_fm_trans),
            #         nu=args.nu,
            #         overwrite='smart'
            #     )

        
        if args.ae : 
            save_results(
            dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb=args.type_emb ,ad_model="AE",
            auc_mean=np.mean(list_auc_ae), ap_mean=np.mean(list_ap_ae),fpr_mean=np.mean(list_fpr_ae),
            auc_std = np.std(list_auc_ae),ap_std =  np.std(list_ap_ae),fpr_std = np.std(list_fpr_ae),
            train_time = np.mean(list_time_ae), nu=args.nu, overwrite='naive'
            )

        if args.rsrae:  
            save_results(
            dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb=args.type_emb ,ad_model="RSRAE",
            auc_mean=np.mean(list_auc_rsrae), ap_mean=np.mean(list_ap_rsrae),fpr_mean=np.mean(list_fpr_rsrae),
            auc_std = np.std(list_auc_rsrae),ap_std =  np.std(list_ap_rsrae),fpr_std = np.std(list_fpr_rsrae),
            train_time = np.mean(list_time_rsrae), nu=args.nu, overwrite='naive'
            )

        if args.ocsvm:
            save_results(
            dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb=args.type_emb ,ad_model="ocsvm",
            auc_mean=np.mean(list_auc_ocsvm), ap_mean=np.mean(list_ap_ocsvm),fpr_mean=np.mean(list_fpr_ocsvm),
            auc_std = np.std(list_auc_ocsvm),ap_std =  np.std(list_ap_ocsvm),fpr_std = np.std(list_fpr_ocsvm),
            train_time = np.mean(list_time_ocsvm), nu=args.nu, overwrite='naive'
            )

        if args.tccm:
            save_results(
            dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb=args.type_emb ,ad_model="TCCM",
            auc_mean=np.mean(list_auc_tccm), ap_mean=np.mean(list_ap_tccm),fpr_mean=np.mean(list_fpr_tccm),
            auc_std = np.std(list_auc_tccm),ap_std =  np.std(list_ap_tccm),fpr_std = np.std(list_fpr_tccm),
            train_time = np.mean(list_time_tccm), nu=args.nu, overwrite='naive'
            )

        if args.cvdd:
            save_results(
            dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb=args.type_emb ,ad_model="CVDD",
            auc_mean=np.mean(list_auc_cvdd), ap_mean=np.mean(list_ap_cvdd),fpr_mean=np.mean(list_fpr_cvdd),
            auc_std = np.std(list_auc_cvdd),ap_std =  np.std(list_ap_cvdd),fpr_std = np.std(list_fpr_cvdd),
            train_time = np.mean(list_time_cvdd), nu=args.nu, overwrite='smart'
            )

        if args.date:
            save_results(
            dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb=args.type_emb ,ad_model="DATE",
            auc_mean=np.mean(list_auc_date), ap_mean=np.mean(list_ap_date),fpr_mean=np.mean(list_fpr_date),
            auc_std = np.std(list_auc_date),ap_std =  np.std(list_ap_date),fpr_std = np.std(list_fpr_date),
            train_time = np.mean(list_time_date), nu=args.nu, overwrite='smart'
            )

        if args.fate:
            save_results(
            dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb=args.type_emb ,ad_model="FATE",
            auc_mean=np.mean(list_auc_fate), ap_mean=np.mean(list_ap_fate),fpr_mean=np.mean(list_fpr_fate),
            auc_std = np.std(list_auc_fate),ap_std =  np.std(list_ap_fate),fpr_std = np.std(list_fpr_fate),
            train_time = np.mean(list_time_fate), nu=args.nu, overwrite='naive'
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments script")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="20newsgroups"
    )

    parser.add_argument(
        "--runall",
        action="store_true"
    )

    args, remaining_argv = parser.parse_known_args()

    if not args.runall:
        parser.add_argument(
        "--inlier_topic",
        type=str,
        default="computer"
        )

    parser.add_argument(
        "--type_emb",
        type=str,
        default="sentence-bert"
    )

    parser.add_argument(
        "--nu",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--nb_runs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--remove_stopwords",
        action="store_true"
    )

    parser.add_argument(
    "--fm",
    action="store_true"
    )

    parser.add_argument(
    "--fm_trans",
    action="store_true"
    )

    parser.add_argument(
    "--ocsvm",
    action="store_true"
    )

    parser.add_argument(
    "--ae",
    action="store_true"
    )

    parser.add_argument(
    "--rsrae",
    action="store_true"
    )   

    parser.add_argument(
    "--tccm",
    action="store_true"
    )  

    parser.add_argument(
    "--cvdd",
    action="store_true"
    )  

    parser.add_argument(
    "--date",
    action="store_true"
    )  
    parser.add_argument(
    "--fate",
    action="store_true"
    )  
    

    args = parser.parse_args()
    main(args)
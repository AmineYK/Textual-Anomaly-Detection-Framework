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
from utils import save_results
import numpy as np
import os
import datasets
from datasets import concatenate_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset_topics_dict= {
    '20newsgroups' : ['computer', 'recreation', 'science', 'miscellaneous', 'politics', 'religion'],
    'reuters' : ['earn', 'trade', 'acq', 'money-fx', 'crude', 'ship', 'interest'],
    'agnews' : ['World', 'Sports', 'Business', 'Sci-Tech'] ,
    'dbpedia14' : ["Company", "Educational Institution", "Artist", "Athlete", "Office Holder", 
                  "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal", "Plant", "Album", "Film", "Written Work"]
}

def main(args):

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


    # for every inlier category 
    for i, inlier_topic in enumerate(inlier_topics):

        print(f"------------------------ {inlier_topic}({i+1}/{len(inlier_topics)}) -----------------------------")

        if args.fm:
            list_auc_fm = []
            list_fpr_fm = []
            list_ap_fm = []
            list_time_fm = []

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

        if args.nu > 0:
            path = os.path.join(save_dir, f"{args.dataset_name}/{inlier_topic}/ds_train_{inlier_topic}_anomaly_{int(args.nu*100)}.pt")
            data_train_anomaly = datasets.load_from_disk(path)
            X_anom_for_train = Tensor(data_train_anomaly['sbert_embeddings']).to(device) 
            print(X_anom_for_train.shape)

        data_train = load_data_inlier(args.dataset_name, inlier_topic, save_dir, is_infec=False, is_cvdd=True)

        if args.fate:
            path = os.path.join(save_dir, f"{args.dataset_name}/{inlier_topic}/ds_train_{inlier_topic}_anomaly.pt")
            data_train_anomaly_fate = datasets.load_from_disk(path)

        if args.type_emb == "fasttext":
            X_inlier = get_data_fasttext(data_train, ft_path, device)
        
        elif args.type_emb == 'sentence-bert':
            X_inlier = Tensor(data_train['sbert_embeddings']).to(device)

        else: raise Exception("type_emb must be completed !")

        print(X_inlier.shape)

        # print(f" Embedding : {args.type_emb} --> {X_inlier.shape}")

        # get the hyperparamter for the FM model for this specific inlier category
        hyp = load_hyperparams(args.dataset_name, inlier_topic, args.type_emb, file_path_hyp)
        # print(hyp)
        nb_runs = 3
        for n_run in range(1, nb_runs):

            print(f"+++++++++++++++++++++ run : {n_run} +++++++++++++++++\n")

            # if args.cvdd:
            #     data_test = load_data_test(args.dataset_name, inlier_topic, n_run, save_dir, is_cvdd=True)
            # X_test, y_test = load_data_test(args.dataset_name, inlier_topic, n_run, save_dir)

            data_test = load_data_test(args.dataset_name, inlier_topic, n_run, save_dir, is_cvdd=True)
            y_test = np.array(data_test['anomaly_class'])

            if args.type_emb == "fasttext":
                X_test = get_data_fasttext(data_test, ft_path, device)
        
            elif args.type_emb == 'sentence-bert':
                X_test = Tensor(data_test['sbert_embeddings']).to(device)

            else: raise Exception("type_emb must be completed !")
            

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
                    _ = ocsvm_model.train(X_inlier.cpu())
                tiic = time.time()
                print(f"\nOCSVM finishing... after {(tiic-taac)/60:.3f} mn")

                auc_ocsvm, fpr95_ocsvm, ap_ocsvm = ocsvm_model.test(X_test.cpu(), y_test)
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
                    "input_dim": X_inlier.shape[1], "hidden_layer_sizes": (64,32,16), "intrinsic_size": 10,
                    "activation": nn.ReLU(), "norm_type": 'l21', "loss_norm_type": 'mse',
                    "if_rsr": True, "enforce_proj": True, "all_alt": True,
                    "learning_rate": 1e-3, "lambda1": 0.1, "lambda2": 0.1,
                    "epoch_size": 20, "batch_show": 50, "normalize": True,
                    "bn": False, "seed": 42, 'batch_size': X_inlier.shape[0] // 5
                }

                rsrae_model = RSRAE(rsrae_args)

                taac = time.time()
                if args.nu > 0.0:
                    _ = rsrae_model.train(torch.concatenate([X_inlier, X_anom_for_train]), device)
                else:
                    _ = rsrae_model.train(X_inlier, device)
                tiic = time.time()
                
                print(f"\nRSRAE finishing... after {(tiic-taac)/60:.3f} mn")

                auc_rsrae, fpr95_rsrae, ap_rsrae = rsrae_model.test(X_test, y_test)
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
                    "epoch_num": 10,
                    "batch_size": X_inlier.shape[0] // 5,
                    "dropout_rate": 0.0,
                    "verbose": 0
                }


                ae_model = AE(ae_args)

                taac = time.time()
                if args.nu > 0:   
                    _ = ae_model.train(torch.concatenate([X_inlier, X_anom_for_train]).cpu())
                else:
                    _ = ae_model.train(X_inlier)
                tiic = time.time()
                print(f"\nAE finishing... after {(tiic-taac)/60:.3f} mn")
                
                auc_ae, fpr95_ae, ap_ae = ae_model.test(X_test, y_test)
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
                    "n_features": X_inlier.shape[1],
                    "epochs" : 30,
                    "learning_rate" : 1e-4,
                    "batch_size": 128,
                    "device": device
                }

                tccm_model = TCCM(tccm_args)

                taac = time.time()
                if args.nu > 0:   
                    _ = tccm_model.train(torch.concatenate([X_inlier, X_anom_for_train]))
                else:
                    _ = tccm_model.train(X_inlier)
                tiic = time.time()
                print(f"\nTCCM finishing... after {(tiic-taac)/60:.3f} mn")
                
                auc_tccm, fpr95_tccm, ap_tccm = tccm_model.test(X_test, y_test)
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
                    "bert_name": "albert-base-v2", #albert-large-v2   
                    "hidden_size": 768, #1024 
                    "n_attention_heads": 10,
                    "attention_size": 150,
                    "freeze_bert": True,
                    "lr": 1e-4,
                    "weight_decay" : 0,
                    "lambda_p": 0.1,
                    "n_epochs": 30,
                    "batch_size": 16,
                    "device": device
                    }
                
                cvdd_model = CVDDModel(cvdd_args)

                taac = time.time()
                if args.nu > 0:
                    cvdd_model.train(concatenate_datasets([data_train, data_train_anomaly]))
                else:
                    cvdd_model.train(data_train)
                tiic = time.time()
                print(f"\CVDD finishing... after {(tiic-taac)/60:.3f} mn")
                
                auc_cvdd, fpr95_cvdd, ap_cvdd = cvdd_model.test(data_test)
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
                    "which_config": "electra",
                    "encoder_name": "google/electra-small-discriminator",
                    "K": 25,
                    "lr": 1e-5,
                    "weight_decay" : 0,
                    "seq_len": 498,
                    "ratio": 0.25,
                    "n_epochs": 20,
                    "batch_size": 32,
                    "device": device
                    }
                            
                date_model = DATEModel(date_args)

                taac = time.time()
                if args.nu > 0:
                    date_model.train(concatenate_datasets([data_train, data_train_anomaly]))
                else:
                    date_model.train(data_train)
                tiic = time.time()
                print(f"\DATE finishing... after {(tiic-taac)/60:.3f} mn")
                
                auc_date, fpr95_date, ap_date = date_model.test(data_test)
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
                    "batch_size": 64,
                    "n_epochs": 3,
                    "lr": 1e-5,
                    "include_regularization": True,
                    "top_k": 0.1,
                    "nb_shot": 10,
                    "train_inlier_text": data_train_['text'],     
                    "train_anomaly_text": data_train_anomaly_fate['text'],
                    "test_inlier_text": data_test_inlier['text'],
                    "test_anomaly_text": data_test_anomaly['text']
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
                    "n_steps": 10
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

        if args.fm:
            save_results(
                dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb=args.type_emb ,ad_model="flow-matching",
                auc_mean=np.mean(list_auc_fm), ap_mean=np.mean(list_ap_fm),fpr_mean=np.mean(list_fpr_fm),
                auc_std = np.std(list_auc_fm),ap_std =  np.std(list_ap_fm),fpr_std = np.std(list_fpr_fm),
                train_time = np.mean(list_time_fm),overwrite='smart'
                )

        
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

        if args.date and nb_runs == 11:
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
            train_time = np.mean(list_time_fate), nu=args.nu, overwrite='smart'
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
    "--fm",
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
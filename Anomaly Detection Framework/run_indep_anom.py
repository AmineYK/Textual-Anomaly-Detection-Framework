import argparse
import logging
import torch
import torch.nn as nn
import time
import tensorflow as tf
import torch.nn.functional as F

from Data_Preparation import utils
from Data_Preparation.Embedding import embedding_encoder
from Modelisation.Baselines.OCSVM.ocsvm import OCSVM
from utils import load_data_inlier, load_data_test, load_hyperparams
from Modelisation.Baselines.RSRAE.model import RSRAE
from Modelisation.Baselines.AE.autoencoder import AE
from Modelisation.Baselines.TCCM.model import TCCM
from Modelisation.Baselines.CVDD.networks.cvdd_Net import CVDDModel
from Modelisation.FlowMatching.flow_matching import BasicFlowMatching
from utils import save_results, create_tables
import numpy as np

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


    # for every inlier category 
    for i, inlier_topic in enumerate(inlier_topics):

        print(f"------------------------ {inlier_topic}({i}/{len(inlier_topics)}) -----------------------------")

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

        # RSRAE --> X_train is infected with some anomalies
        if args.rsrae:
            X_inlier_anoma = load_data_inlier(args.dataset_name, inlier_topic, save_dir, True)
        
        if args.cvdd:
            data_train = load_data_inlier(args.dataset_name, inlier_topic, save_dir, is_infec=False, is_cvdd=True)
            
        # load the X_inlier matrix
        X_inlier = load_data_inlier(args.dataset_name, inlier_topic, save_dir)


        # get the hyperparamter for the FM model for this specific inlier category
        hyp = load_hyperparams(args.dataset_name, inlier_topic, file_path_hyp)
        print(hyp)

        for n_run in range(1,11):

            print(f"+++++++++++++++++++++ run : {n_run} +++++++++++++++++\n")

            if args.cvdd:
                data_test = load_data_test(args.dataset_name, inlier_topic, n_run, save_dir, is_cvdd=True)
            X_test, y_test = load_data_test(args.dataset_name, inlier_topic, n_run, save_dir)

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
                    "bn": False, "seed": 42, 'batch_size': X_inlier.shape[0] // 100
                }

                rsrae_model = RSRAE(rsrae_args)

                taac = time.time()
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
                    "batch_size": X_inlier.shape[0] // 100,
                    "dropout_rate": 0.0,
                    "verbose": 0
                }


                ae_model = AE(ae_args)

                taac = time.time()
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
                    "epochs" : 50,
                    "learning_rate" : 1e-3,
                    "batch_size": 64,
                    "device": device
                }

                tccm_model = TCCM(tccm_args)

                taac = time.time()
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

            if args.cvdd:
                cvdd_args = {
                    "type_emb": "fasttext",
                    # "emb_model": "distilbert-base-uncased",
                    "emb_model": "distilroberta-base",
                    "attention_size": 150,
                    "n_attention_heads": 10,
                    "lr": 0.001,
                    "weight_decay": 0,
                    "lr_milestones": (10, 25),
                    "n_epochs": 30,
                    "lambda_p": 1.0,
                    "alpha_scheduler": "logarithmic",
                    "seq_len": 100,
                    "batch_size": 64,
                    "min_freq": 1,
                    "device": device
                }

                cvdd_model = CVDDModel(cvdd_args)

                taac = time.time()
                cvdd_model_trained, cvdd_trainer = cvdd_model.train(data_train, data_test)
                tiic = time.time()
                print(f"\CVDD finishing... after {(tiic-taac)/60:.3f} mn")
                
                auc_cvdd, fpr95_cvdd, ap_cvdd = cvdd_model.test(cvdd_model_trained, cvdd_trainer, data_train, data_test)
                print(f"CVDD --> AUC: {auc_cvdd:.4f} | FPR@95: {fpr95_cvdd:.4f} | AP: {ap_cvdd:.4f}\n")
                
                list_auc_cvdd.append(auc_cvdd)
                list_fpr_cvdd.append(fpr95_cvdd)    
                list_ap_cvdd.append(ap_cvdd)  
                list_time_cvdd.append((tiic-taac))  


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
        
                list_auc_fm.append(auc_fm)
                list_fpr_fm.append(fpr95_fm)    
                list_ap_fm.append(ap_fm)
                list_time_fm.append((tiic-taac))


        # print("AE --> ",inlier_topic ,np.mean(list_auc_ae))
        # print(inlier_topic ,np.mean(list_auc_fm))
        # print("RSRAE --> ", inlier_topic ,np.mean(list_auc_rsrae))
        print("CVDD --> ", inlier_topic ,np.mean(list_auc_cvdd))
        if args.fm:
            save_results(
                dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb="sentence_bert" ,ad_model="flow-matching",
                auc_mean=np.mean(list_auc_fm), ap_mean=np.mean(list_ap_fm),fpr_mean=np.mean(list_fpr_fm),
                auc_std = np.std(list_auc_fm),ap_std =  np.std(list_ap_fm),fpr_std = np.std(list_fpr_fm),
                train_time = np.mean(list_time_fm),overwrite='smart'
                )

        
        if args.ae : 
            save_results(
            dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb="sentence_bert" ,ad_model="AE",
            auc_mean=np.mean(list_auc_ae), ap_mean=np.mean(list_ap_ae),fpr_mean=np.mean(list_fpr_ae),
            auc_std = np.std(list_auc_ae),ap_std =  np.std(list_ap_ae),fpr_std = np.std(list_fpr_ae),
            train_time = np.mean(list_time_ae),overwrite='naive'
            )

        if args.rsrae:  
            save_results(
            dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb="sentence_bert" ,ad_model="RSRAE",
            auc_mean=np.mean(list_auc_rsrae), ap_mean=np.mean(list_ap_rsrae),fpr_mean=np.mean(list_fpr_rsrae),
            auc_std = np.std(list_auc_rsrae),ap_std =  np.std(list_ap_rsrae),fpr_std = np.std(list_fpr_rsrae),
            train_time = np.mean(list_time_rsrae), overwrite='smart'
            )

        if args.ocsvm:
            save_results(
            dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb="sentence_bert" ,ad_model="ocsvm",
            auc_mean=np.mean(list_auc_ocsvm), ap_mean=np.mean(list_ap_ocsvm),fpr_mean=np.mean(list_fpr_ocsvm),
            auc_std = np.std(list_auc_ocsvm),ap_std =  np.std(list_ap_ocsvm),fpr_std = np.std(list_fpr_ocsvm),
            train_time = np.mean(list_time_ocsvm), overwrite='naive'
            )

        if args.tccm:
            save_results(
            dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb="sentence_bert" ,ad_model="TCCM",
            auc_mean=np.mean(list_auc_tccm), ap_mean=np.mean(list_ap_tccm),fpr_mean=np.mean(list_fpr_tccm),
            auc_std = np.std(list_auc_tccm),ap_std =  np.std(list_ap_tccm),fpr_std = np.std(list_fpr_tccm),
            train_time = np.mean(list_time_tccm), overwrite='naive'
            )

        if args.cvdd:
            save_results(
            dataset_name=args.dataset_name, inlier_topic=inlier_topic ,type_emb=cvdd_args['type_emb'] ,ad_model="CVDD",
            auc_mean=np.mean(list_auc_cvdd), ap_mean=np.mean(list_ap_cvdd),fpr_mean=np.mean(list_fpr_cvdd),
            auc_std = np.std(list_auc_cvdd),ap_std =  np.std(list_ap_cvdd),fpr_std = np.std(list_fpr_cvdd),
            train_time = np.mean(list_time_cvdd), overwrite='smart'
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
    

    args = parser.parse_args()
    main(args)
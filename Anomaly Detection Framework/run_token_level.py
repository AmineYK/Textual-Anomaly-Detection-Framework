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
from Modelisation.Baselines.CVDD.networks.model_bert import CVDDModel
from Modelisation.Baselines.FATE.fate import FATEModel
from Modelisation.Baselines.DATE.date import DATEModel
from Modelisation.FlowMatching.flow_matching_transformers_toksen import FlowDiTTokSen, FlowMatchingTransformersTokSen
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

dataset_topics_dict = {
    '20newsgroups': ['computer', 'recreation', 'science', 'miscellaneous', 'politics', 'religion'],
    'reuters':      ['earn', 'trade', 'acq', 'money-fx', 'crude', 'ship', 'interest'],
    'agnews':       ['World', 'Sports', 'Business', 'Sci-Tech'],
    'dbpedia14':    ["Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
                     "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal",
                     "Plant", "Album", "Film", "Written Work"],
    'sms':          ['normal'],
    'enron':        ['normal'],
    'imdb':         ['positive', 'negative'],
    'sst2':         ['positive', 'negative'],
    'mage':         ['normal'],
    'm4':           ["wikipedia", "arxiv", "wikihow", "reddit", "peerread"],
}

DATASET_COL = {
    'dbpedia14': 'content',
}
DEFAULT_COL = 'text'

TOKEN_MODEL_CONFIGS = {
    'roberta': {
        'model_name': 'roberta-base',
        'model_type': 'encoder',
    },
    'modernbert': {
        'model_name': 'answerdotai/ModernBERT-base',
        'model_type': 'encoder',
    },
    'qwen': {
        'model_name': 'Qwen/Qwen2.5-1.5B',
        'model_type': 'decoder',
    },
    'mistral': {
        'model_name': 'ministral/Ministral-3b-instruct',
        'model_type': 'decoder',
    },
}

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    len_seq = 128

    # topics à traiter
    if args.runall:
        inlier_topics = dataset_topics_dict[args.dataset_name]
    else:
        inlier_topics = [args.inlier_topic]

    # colonne texte selon le dataset
    COL = DATASET_COL.get(args.dataset_name, DEFAULT_COL)

    save_dir = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Data"
    print(f"\n<<<<<<<<<<<<<< {args.dataset_name} >>>>>>>>>>>>>>>>>>>\n")
    print(f"Text column : '{COL}'")

    if args.type_emb in TOKEN_MODEL_CONFIGS:
        cfg = TOKEN_MODEL_CONFIGS[args.type_emb]
        print(f"Loading token model : {cfg['model_name']} (type={cfg['model_type']})")
        tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
        bertmodel = AutoModel.from_pretrained(
            cfg['model_name'],
            torch_dtype=torch.float16 if cfg['model_type'] == 'decoder' else torch.float32
        ).to(device)
        bertmodel.eval()
    else:
        raise ValueError(f"type_emb '{args.type_emb}' non reconnu. Choisir parmi : {list(TOKEN_MODEL_CONFIGS.keys())}")

    for i, inlier_topic in enumerate(inlier_topics):

        print(f"------------------------ {inlier_topic}({i+1}/{len(inlier_topics)}) -----------------------------")

        if args.fm_trans:
            list_auc_fm_trans, list_fpr_fm_trans, list_ap_fm_trans, list_time_fm_trans = [], [], [], []

        if args.ocsvm:
            list_auc_ocsvm, list_fpr_ocsvm, list_ap_ocsvm, list_time_ocsvm = [], [], [], []

        if args.rsrae:
            list_auc_rsrae, list_fpr_rsrae, list_ap_rsrae, list_time_rsrae = [], [], [], []

        if args.ae:
            list_auc_ae, list_fpr_ae, list_ap_ae, list_time_ae = [], [], [], []

        if args.tccm:
            list_auc_tccm, list_fpr_tccm, list_ap_tccm, list_time_tccm = [], [], [], []

        if args.cvdd:
            list_auc_cvdd, list_fpr_cvdd, list_ap_cvdd, list_time_cvdd = [], [], [], []

        if args.date:
            list_auc_date, list_fpr_date, list_ap_date, list_time_date = [], [], [], []

        if args.fate:
            list_auc_fate, list_fpr_fate, list_ap_fate, list_time_fate = [], [], [], []

        # ── Chargement données train ───────────────────────────
        data_train = load_data_inlier(args.dataset_name, inlier_topic, save_dir, is_infec=False, is_cvdd=True)

        if args.fate:
            path = os.path.join(save_dir, f"{args.dataset_name}/{inlier_topic}/ds_train_{inlier_topic}_anomaly.pt")
            data_train_anomaly_fate = datasets.load_from_disk(path)

        # ── Encodage train ─────────────────────────────────────
        X_inlier, _, attentions_train_mask = encode_tokens(
            bertmodel, tokenizer, data_train[COL], device,
            batch_size=64,
            max_length=len_seq,
            model_type=cfg['model_type']
        )

        if args.nu > 0:
            path = os.path.join(save_dir, f"{args.dataset_name}/{inlier_topic}/ds_train_{inlier_topic}_anomaly_{int(args.nu*100)}.pt")
            data_train_anomaly = datasets.load_from_disk(path)
            X_anom_for_train = Tensor(data_train_anomaly['sbert_embeddings']).to(device)
            print(X_anom_for_train.shape)

        print(X_inlier.shape)

        for n_run in range(1, args.nb_runs):

            print(f"+++++++++++++++++++++ run : {n_run} / {args.nb_runs - 1} +++++++++++++++++\n")

            data_test = load_data_test(args.dataset_name, inlier_topic, n_run, save_dir, is_cvdd=True)
            y_test = np.array(data_test['anomaly_class'])

            # ── Encodage test ──────────────────────────────────
            X_test, _, attentions_test_mask = encode_tokens(
                bertmodel, tokenizer, data_test[COL], device,
                batch_size=64,
                max_length=len_seq,
                model_type=cfg['model_type']
            )

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
                    _ = ocsvm_model.train(X_inlier.mean(dim=1).cpu())
                tiic = time.time()
                print(f"\nOCSVM finishing... after {(tiic-taac)/60:.3f} mn")

                auc_ocsvm, fpr95_ocsvm, ap_ocsvm = ocsvm_model.test(X_test.mean(dim=1).cpu(), y_test)
                print(f"OCSVM --> AUC: {auc_ocsvm:.4f} | FPR@95: {fpr95_ocsvm:.4f} | AP: {ap_ocsvm:.4f}\n")

                list_auc_ocsvm.append(auc_ocsvm)
                list_fpr_ocsvm.append(fpr95_ocsvm)
                list_ap_ocsvm.append(ap_ocsvm)
                list_time_ocsvm.append((tiic-taac))

            #########################################
            ################# RSRAE #################
            #########################################

            if args.rsrae:
                rsrae_args = {
                    "input_dim": X_inlier.shape[2], "hidden_layer_sizes": (128, 64, 32), "intrinsic_size": 10,
                    "activation": nn.ReLU(), "norm_type": 'l21', "loss_norm_type": 'mse',
                    "if_rsr": True, "enforce_proj": True, "all_alt": True,
                    "learning_rate": 1e-4, "lambda1": 0.1, "lambda2": 0.1,
                    "epoch_size": 100, "batch_show": 50, "normalize": True,
                    "bn": False, "seed": 42, 'batch_size': X_inlier.shape[0] // 100
                }

                rsrae_model = RSRAE(rsrae_args)

                taac = time.time()
                if args.nu > 0.0:
                    _ = rsrae_model.train(torch.concatenate([X_inlier.mean(dim=1), X_anom_for_train.mean(dim=1)]), device)
                else:
                    _ = rsrae_model.train(X_inlier.mean(dim=1), device)
                tiic = time.time()
                print(f"\nRSRAE finishing... after {(tiic-taac)/60:.3f} mn")

                auc_rsrae, fpr95_rsrae, ap_rsrae = rsrae_model.test(X_test.mean(dim=1), y_test, device)
                print(f"RSRAE --> AUC: {auc_rsrae:.4f} | FPR@95: {fpr95_rsrae:.4f} | AP: {ap_rsrae:.4f}\n")

                list_auc_rsrae.append(auc_rsrae)
                list_fpr_rsrae.append(fpr95_rsrae)
                list_ap_rsrae.append(ap_rsrae)
                list_time_rsrae.append((tiic-taac))

            ######################################
            ################# AE #################
            ######################################

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
                    _ = ae_model.train(X_inlier.mean(dim=1))
                tiic = time.time()
                print(f"\nAE finishing... after {(tiic-taac)/60:.3f} mn")

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
                tccm_args = {
                    "n_features": X_inlier.shape[2],
                    "epochs": 50,
                    "learning_rate": 1e-3,
                    "batch_size": 256,
                    "device": device
                }

                tccm_model = TCCM(tccm_args)

                taac = time.time()
                if args.nu > 0:
                    _ = tccm_model.train(torch.concatenate([X_inlier, X_anom_for_train]))
                else:
                    _ = tccm_model.train(X_inlier.mean(dim=1))
                tiic = time.time()
                print(f"\nTCCM finishing... after {(tiic-taac)/60:.3f} mn")

                auc_tccm, fpr95_tccm, ap_tccm = tccm_model.test(X_test.mean(dim=1), y_test)
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
                    "bert_name": "roberta-base",
                    "hidden_size": 768,
                    "n_attention_heads": 10,
                    "attention_size": 64,
                    "freeze_bert": True,
                    "lr": 1e-3,
                    "weight_decay": 0,
                    "lambda_p": 0.1,
                    "n_epochs": 30,
                    "batch_size": 256,
                    "device": device
                }

                cvdd_model = CVDDModel(cvdd_args)

                taac = time.time()
                if args.nu > 0:
                    cvdd_model.train(concatenate_datasets([data_train, data_train_anomaly]))
                else:
                    cvdd_model.train(data_train, COL)
                tiic = time.time()
                print(f"\nCVDD finishing... after {(tiic-taac)/60:.3f} mn")

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
                    "which_config": "roberta",
                    "encoder_name": "roberta-base",
                    "K": 20,
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "seq_len": 256,
                    "ratio": 0.50,
                    "n_epochs": 50,
                    "batch_size": 256,
                    "device": device
                }

                date_model = DATEModel(date_args)

                taac = time.time()
                if args.nu > 0:
                    date_model.train(concatenate_datasets([data_train, data_train_anomaly]))
                else:
                    date_model.train(data_train, COL)
                tiic = time.time()
                print(f"\nDATE finishing... after {(tiic-taac)/60:.3f} mn")

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

                data_train_ = concatenate_datasets([data_train, data_train_anomaly]) if args.nu > 0 else data_train

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
                print(f"\nFATE finishing... after {(tiic-taac)/60:.3f} mn")

                auc_fate, fpr95_fate, ap_fate = fate_model.test()
                print(f"FATE --> AUC: {auc_fate:.4f} | FPR@95: {fpr95_fate:.4f} | AP: {ap_fate:.4f}\n")

                list_auc_fate.append(auc_fate)
                list_fpr_fate.append(fpr95_fate)
                list_ap_fate.append(ap_fate)
                list_time_fate.append((tiic-taac))

            ######################################################
            ############# Flow Matching Transformers #############
            ######################################################

            if args.fm_trans:
                fm_trans_config = {
                    'latent_dim': 768,
                    'hidden_dim': 128,
                    'depth': 8,
                    'n_heads': 8,
                    'freq_embed_size': 128,
                    'lr': 1e-3,
                    'weight_decay': 1e-5,
                    'lambda_svdd': 1e-2,
                    'epochs': 350,
                    'lr_epochs': 150,
                    'batch_size': 512,
                    'coef_var': 1,
                    'target': 'gaussian-neigh',
                    'source': X_inlier,
                    'attentions_mask': attentions_train_mask,
                    'device': device
                }

                flowmodel = FlowDiTTokSen(
                    latent_dim=fm_trans_config['latent_dim'],
                    hidden_dim=fm_trans_config['hidden_dim'],
                    depth=fm_trans_config['depth'],
                    n_heads=fm_trans_config['n_heads']
                ).to(device)

                fm_transformer = FlowMatchingTransformersTokSen(flowmodel, fm_trans_config)

                taac = time.time()
                fm_transformer.train(True)
                tiic = time.time()
                print(f"\nFM Transformer finishing... after {(tiic-taac)/60:.3f} mn")

                auc_fm_trans, fpr95_fm_trans, ap_fm_trans = fm_transformer.test(X_test, y_test, attentions_test_mask, type='norm-centroid', n_steps=10)
                print(f"FM --> AUC: {auc_fm_trans:.4f} | FPR@95: {fpr95_fm_trans:.4f} | AP: {ap_fm_trans:.4f}\n")

                list_auc_fm_trans.append(auc_fm_trans)
                list_fpr_fm_trans.append(fpr95_fm_trans)
                list_ap_fm_trans.append(ap_fm_trans)
                list_time_fm_trans.append((tiic-taac))


        if args.fm_trans:
            print(inlier_topic, np.mean(list_auc_fm_trans), np.mean(list_fpr_fm_trans), np.mean(list_ap_fm_trans))
            save_results(
                dataset_name=args.dataset_name, inlier_topic=inlier_topic, type_emb=args.type_emb, ad_model="flow-matching-Transformers-Comp",
                auc_mean=np.mean(list_auc_fm_trans), ap_mean=np.mean(list_ap_fm_trans), fpr_mean=np.mean(list_fpr_fm_trans),
                auc_std=np.std(list_auc_fm_trans), ap_std=np.std(list_ap_fm_trans), fpr_std=np.std(list_fpr_fm_trans),
                train_time=np.mean(list_time_fm_trans), nu=args.nu, overwrite='naive'
            )

        if args.ae:
            save_results(
                dataset_name=args.dataset_name, inlier_topic=inlier_topic, type_emb=args.type_emb, ad_model="AE",
                auc_mean=np.mean(list_auc_ae), ap_mean=np.mean(list_ap_ae), fpr_mean=np.mean(list_fpr_ae),
                auc_std=np.std(list_auc_ae), ap_std=np.std(list_ap_ae), fpr_std=np.std(list_fpr_ae),
                train_time=np.mean(list_time_ae), nu=args.nu, overwrite='naive'
            )

        if args.rsrae:
            save_results(
                dataset_name=args.dataset_name, inlier_topic=inlier_topic, type_emb=args.type_emb, ad_model="RSRAE",
                auc_mean=np.mean(list_auc_rsrae), ap_mean=np.mean(list_ap_rsrae), fpr_mean=np.mean(list_fpr_rsrae),
                auc_std=np.std(list_auc_rsrae), ap_std=np.std(list_ap_rsrae), fpr_std=np.std(list_fpr_rsrae),
                train_time=np.mean(list_time_rsrae), nu=args.nu, overwrite='naive'
            )

        if args.ocsvm:
            save_results(
                dataset_name=args.dataset_name, inlier_topic=inlier_topic, type_emb=args.type_emb, ad_model="ocsvm",
                auc_mean=np.mean(list_auc_ocsvm), ap_mean=np.mean(list_ap_ocsvm), fpr_mean=np.mean(list_fpr_ocsvm),
                auc_std=np.std(list_auc_ocsvm), ap_std=np.std(list_ap_ocsvm), fpr_std=np.std(list_fpr_ocsvm),
                train_time=np.mean(list_time_ocsvm), nu=args.nu, overwrite='naive'
            )

        if args.tccm:
            save_results(
                dataset_name=args.dataset_name, inlier_topic=inlier_topic, type_emb=args.type_emb, ad_model="TCCM",
                auc_mean=np.mean(list_auc_tccm), ap_mean=np.mean(list_ap_tccm), fpr_mean=np.mean(list_fpr_tccm),
                auc_std=np.std(list_auc_tccm), ap_std=np.std(list_ap_tccm), fpr_std=np.std(list_fpr_tccm),
                train_time=np.mean(list_time_tccm), nu=args.nu, overwrite='naive'
            )

        if args.cvdd:
            save_results(
                dataset_name=args.dataset_name, inlier_topic=inlier_topic, type_emb=args.type_emb, ad_model="CVDD",
                auc_mean=np.mean(list_auc_cvdd), ap_mean=np.mean(list_ap_cvdd), fpr_mean=np.mean(list_fpr_cvdd),
                auc_std=np.std(list_auc_cvdd), ap_std=np.std(list_ap_cvdd), fpr_std=np.std(list_fpr_cvdd),
                train_time=np.mean(list_time_cvdd), nu=args.nu, overwrite='naive'
            )

        if args.date:
            save_results(
                dataset_name=args.dataset_name, inlier_topic=inlier_topic, type_emb=args.type_emb, ad_model="DATE",
                auc_mean=np.mean(list_auc_date), ap_mean=np.mean(list_ap_date), fpr_mean=np.mean(list_fpr_date),
                auc_std=np.std(list_auc_date), ap_std=np.std(list_ap_date), fpr_std=np.std(list_fpr_date),
                train_time=np.mean(list_time_date), nu=args.nu, overwrite='naive'
            )

        if args.fate:
            save_results(
                dataset_name=args.dataset_name, inlier_topic=inlier_topic, type_emb=args.type_emb, ad_model="FATE",
                auc_mean=np.mean(list_auc_fate), ap_mean=np.mean(list_ap_fate), fpr_mean=np.mean(list_fpr_fate),
                auc_std=np.std(list_auc_fate), ap_std=np.std(list_ap_fate), fpr_std=np.std(list_fpr_fate),
                train_time=np.mean(list_time_fate), nu=args.nu, overwrite='naive'
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments script")

    parser.add_argument("--dataset_name", type=str, default="20newsgroups")
    parser.add_argument("--runall", action="store_true")

    args, remaining_argv = parser.parse_known_args()

    if not args.runall:
        parser.add_argument("--inlier_topic", type=str, default="computer")

    parser.add_argument(
        "--type_emb", type=str, default="roberta",
        choices=list(TOKEN_MODEL_CONFIGS.keys()),
        help=f"Modèle token-level à utiliser : {list(TOKEN_MODEL_CONFIGS.keys())}"
    )
    parser.add_argument("--nu",      type=float, default=0.0)
    parser.add_argument("--nb_runs", type=int,   default=5)

    parser.add_argument("--fm",       action="store_true")
    parser.add_argument("--fm_trans", action="store_true")
    parser.add_argument("--ocsvm",    action="store_true")
    parser.add_argument("--ae",       action="store_true")
    parser.add_argument("--rsrae",    action="store_true")
    parser.add_argument("--tccm",     action="store_true")
    parser.add_argument("--cvdd",     action="store_true")
    parser.add_argument("--date",     action="store_true")
    parser.add_argument("--fate",     action="store_true")

    args = parser.parse_args()
    main(args)
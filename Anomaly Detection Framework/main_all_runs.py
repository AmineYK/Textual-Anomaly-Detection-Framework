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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re
import os

def save_results(args, auc_mean, ap_mean, fpr_mean, auc_std=None, ap_std=None, fpr_std=None,
                 output_dir="/home/youcefk251/My Thesis/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results",
                 filename="results.txt",
                 overwrite=None):  # overwrite: "naive", "smart", or None

    import os, re

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    existing_content = ""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing_content = f.read()

    # --- Pattern unique d’identification du bloc ---
    pattern = (
        rf"Dataset:\s*{re.escape(args.dataset_name)}\s*"
        rf"Inlier class:\s*{re.escape(args.inlier_topic)}\s*"
        rf"Embedding type:\s*{re.escape(args.type_emb)}\s*"
        rf"AD model:\s*{re.escape(args.ad_model)}"
    )

    # --- Bloc de texte à écrire ---
    def fmt(mean, std):
        """Formate 'mean ± std' si std est fourni, sinon seulement mean."""
        return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"

    new_block = (
        "========================================\n"
        f"Dataset:        {args.dataset_name}\n"
        f"Inlier class:   {args.inlier_topic}\n"
        f"Embedding type: {args.type_emb}\n"
        f"AD model:       {args.ad_model}\n"
        "----------------------------------------\n"
        f"AUC:            {fmt(auc_mean, auc_std)}\n"
        f"Avg Precision:  {fmt(ap_mean, ap_std)}\n"
        f"FPR@95:         {fmt(fpr_mean, fpr_std)}\n"
        "========================================\n\n"
    )

    # --- Recherche si un bloc identique existe déjà ---
    match = re.search(pattern, existing_content)

    if match:
        # Repère l'ancien bloc complet
        old_block_pattern = (
            r"========================================\n"
            + pattern +
            r".*?========================================\n\n"
        )
        old_block = re.search(old_block_pattern, existing_content, flags=re.DOTALL)
        if old_block:
            old_block = old_block.group(0)
            old_auc_match = re.search(r"AUC:\s*([\d.]+)", old_block)
            old_auc = float(old_auc_match.group(1)) if old_auc_match else -1
        else:
            old_auc = -1

        # --- Logique de remplacement ---
        do_replace = False
        if overwrite == "naive":
            do_replace = True
        elif overwrite == "smart" and auc_mean > old_auc:
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

    # --- Écriture dans le fichier ---
    with open(filepath, "w") as f:
        f.write(existing_content)



def run_multiple_times(args, n_runs=10):
    """Exécute n_runs fois le modèle et retourne les moyennes et écarts-types."""
    aucs, aps, fprs = [], [], []

    for i in range(n_runs):
        print(f"\n===== Run {i + 1}/{n_runs} for model {args.ad_model} =====")

        # --- Refaire la préparation des données à chaque run ---
        required_encoding = args.ad_model == 'ocsvm'
        dp_dict = data_preparation(args, logger, embedding_encoding=required_encoding)

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

        # === OCSVM ===
        if args.ad_model == 'ocsvm':
            ocsvm_kwargs = {"nu": args.nu, "kernel": args.kernel, "gamma": args.gamma}
            clf, _, _ = ocsvm.One_Class_SVM(data_train.inputs, ocsvm_kwargs)
            scores_test = clf.decision_function(data_test.inputs)
            auc, ap, fpr95 = ev.evaluation(data_test.labels, scores_test, verbose=False)

        # === CVDD ===
        elif args.ad_model == 'cvdd':
            if args.type_emb == 'bert':
                tokenizer = AutoTokenizer.from_pretrained(args.emb_model)
                vocab = None
            elif args.type_emb in ('glove', 'fasttext'):
                corpus = data_train['text']
                vocab = utils.build_vocab(corpus, min_freq=1)
                tokenizer = None

            model, dl_train, dl_test = utils.cvdd_model_pipeline(
                data_train, data_test, args.attention_size, args.n_attention_heads,
                args.type_emb, 500, args.batch_size, args.shuffle, tokenizer, vocab
            )

            cvdd_trainer = cvdd_Net.CVDDTrainer(
                optimizer_name='adam', learning_rate=args.lr, lr_milestones=(args.lr_milestones[0], args.lr_milestones[1]),
                n_epochs=args.n_epochs, lambda_p=args.lambda_p,
                alpha_scheduler=args.alpha_scheduler, weight_decay=1e-4
            )

            model_trained = cvdd_trainer.train(model, dl_train)
            auc, ap, fpr95, _ = cvdd_trainer.test(model_trained, dl_test, ad_score='context_dist_mean')

        aucs.append(auc)
        aps.append(ap)
        fprs.append(fpr95)

    # Moyenne et écart-type
    auc_mean, auc_std = np.mean(aucs), np.std(aucs)
    ap_mean, ap_std = np.mean(aps), np.std(aps)
    fpr_mean, fpr_std = np.mean(fprs), np.std(fprs)

    return auc_mean, ap_mean, fpr_mean, auc_std, ap_std, fpr_std


def main(args):

    # Vérifications
    if args.ad_model in ['ocsvm', 'cvdd'] and args.training_mode == 'two_classes':
        raise Exception(f"Warning ! the 'training_mode' : '{args.training_mode}' is not possible with '{args.ad_model}' model")

    logger.info(
        f"\nStarting 10-run execution with dataset='{args.dataset_name}', "
        f"inlier_topic='{args.inlier_topic}', model='{args.ad_model}', emb='{args.type_emb}'.\n"
    )

    start = time.time()
    n_runs = 3

    if args.ad_model in ['ocsvm', 'cvdd']:
        auc_mean, ap_mean, fpr_mean, auc_std, ap_std, fpr_std = run_multiple_times(args, n_runs=n_runs)

        logger.info(
            f"\nResults over {n_runs} runs:\n"
            f"AUC  = {auc_mean:.4f} ± {auc_std:.4f}\n"
            f"AP   = {ap_mean:.4f} ± {ap_std:.4f}\n"
            f"FPR@95 = {fpr_mean:.4f} ± {fpr_std:.4f}\n"
        )

        save_results(
            args,
            auc_mean,ap_mean,
            fpr_mean, auc_std,
            ap_std, fpr_std,
            output_dir="/home/youcefk251/My Thesis/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results",
            filename="results.txt",
            overwrite="smart"
        )

    else:
        print(f"Model '{args.ad_model}' not recognized for multi-run execution.")

    end = time.time()
    logger.info(f"Full execution (10 runs) finished after {end - start:.2f} seconds.")



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
        parser.add_argument("--gamma", type=float, default=1, help="OCSVM gamme parameter")

    parser.add_argument(
        "--lambda_p",
        type=float,
        default=1.0,
        help="Lmabda_p"
    )

    parser.add_argument(
        "--alpha_scheduler",
        type=str,
        default="logarithmic",
        help="scheduler"
    )

    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )   

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning_rate"
    )   

    parser.add_argument(
        "--lr_milestones",
        type=int,
        nargs='+',
        default=[40, 60],
        help="lr_milestones"
    )   
    args = parser.parse_args()
    main(args)

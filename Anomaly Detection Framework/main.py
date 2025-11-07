import argparse
import logging
import time
from transformers import AutoTokenizer
from Modelisation.Baselines.OCSVM import ocsvm
import Modelisation.evaluation as ev
from Modelisation.Baselines.CVDD.utils import build_vocab, cvdd_model_pipeline
from Modelisation.Baselines.CVDD.networks import cvdd_Net
from Data_Preparation.utils import data_preparation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re
import os

def save_results(args, auc_mean, ap_mean, fpr_mean, auc_std=None, ap_std=None, fpr_std=None,
                 output_dir="/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results",
                 filename="results.txt",
                 overwrite=None): 

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    existing_content = ""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing_content = f.read()

  
    pattern = (
        rf"Dataset:\s*{re.escape(args.dataset_name)}\s*"
        rf"Inlier class:\s*{re.escape(args.inlier_topic)}\s*"
        rf"Embedding type:\s*{re.escape(args.type_emb)}\s*"
        rf"AD model:\s*{re.escape(args.ad_model)}"
    )


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

    match = re.search(pattern, existing_content)

    if match:
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

    with open(filepath, "w") as f:
        f.write(existing_content)


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
        clf, _, _ = ocsvm.One_Class_SVM(data_train.inputs, ocsvm_kwargs)

        _ = clf.predict(data_test.inputs)           
        scores_test = clf.decision_function(data_test.inputs)

        auc, ap, fpr95 = ev.evaluation(data_test.labels, scores_test, verbose=False)

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
            vocab = build_vocab(corpus,min_freq=1)
            tokenizer = None

        model, dl_train, dl_test = cvdd_model_pipeline(data_train, data_test, args.attention_size, args.n_attention_heads, 
                                                       args.type_emb, 500, args.batch_size, args.shuffle, tokenizer, vocab)

        cvdd_trainer = cvdd_Net.CVDDTrainer(optimizer_name='adam', learning_rate=args.lr, lr_milestones=(args.lr_milestones[0], args.lr_milestones[1]),
                                            n_epochs=args.n_epochs, lambda_p=args.lambda_p,
                                            alpha_scheduler=args.alpha_scheduler, weight_decay=1e-4, device=args.device)
        logger.info('################################')
        logger.info('Model Training...')
        logger.info('################################')
        
        model_trained = cvdd_trainer.train(model, dl_train)
        
        logger.info('################################')
        logger.info(f'Model Testing ...')
        logger.info('################################')
        auc, ap, fpr95, _ = cvdd_trainer.test(model_trained, dl_test, ad_score='context_dist_mean')

        save_results(args, auc, ap, fpr95,
                 output_dir="/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results",
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
        "--device",
        type=str,
        default="cuda",
        help="The type of device"
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
    elif args.ad_model == "ocsvm":
        
        parser.add_argument("--nu", type=float, default=0.1, help="OCSVM nu parameter")
        parser.add_argument("--kernel", type=str, default="rbf", help="OCSVM kernel")
        parser.add_argument("--gamma", type=float, default=1, help="OCSVM gamme parameter")

    args = parser.parse_args()
    main(args)

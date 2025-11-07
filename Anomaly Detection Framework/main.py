import argparse
import logging
import time
from transformers import AutoTokenizer
from Modelisation.Baselines.OCSVM import ocsvm
import Modelisation.evaluation as ev
from Modelisation.Baselines.CVDD.utils import build_vocab, cvdd_model_pipeline
from Modelisation.Baselines.CVDD.networks import cvdd_Net
from Data_Preparation.utils import data_preparation
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        utils.save_results(args, auc, ap, fpr95,
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

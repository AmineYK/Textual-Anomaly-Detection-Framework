from Data_Preparation.Dataset import ADdatasets
from Data_Preparation.Tac import tac
from Data_Preparation.Embedding import embedding_encoder
import argparse
import logging
from torch.utils.data import DataLoader
import time
from Modelisation.Baselines.OCSVM import ocsvm
import Modelisation.evaluation as ev


# --- Configuration du logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_preparation(args, logger, ADdatasets, tac, embedding_encoder):

    logger.info("################################")
    logger.info("Loading Dataset...")
    logger.info("################################\n")

    dataset = ADdatasets.ADDataset(args.dataset_name, args.full_dataset_, args.preprocessing)

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
        inlier_dataset_train, anomaly_dataset_train = tac.textual_anomaly_contamination(
            dataset_train, args.dataset_name, args.inlier_topic, args.type_tac, args.anomaly_rate
        )
        inlier_dataset_test, anomaly_dataset_test = tac.textual_anomaly_contamination(
            dataset_test, args.dataset_name, args.inlier_topic, args.type_tac, args.anomaly_rate
        )
    else:
        inlier_dataset_complet, anomaly_dataset_complet = tac.textual_anomaly_contamination(
            dataset_complet, args.dataset_name, args.inlier_topic, args.type_tac, args.anomaly_rate
        )

    logger.info("################################")
    logger.info("Embedding Encodage...")
    logger.info("#################################\n")

    emb_encoder = embedding_encoder.EmbeddingEncoder(args.model_name, args.type_emb)

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
        wrapper_inlier_complet = ADdatasets.DatasetWrapper(inlier_dataset_complet_emb, args.type_emb)
        wrapper_anomaly_complet = ADdatasets.DatasetWrapper(anomaly_dataset_complet_emb, args.type_emb)

        inlier_dataloader = DataLoader(wrapper_inlier_complet, batch_size=args.batch_size, shuffle=args.shuffle)
        anomaly_dataloader = DataLoader(wrapper_anomaly_complet, batch_size=args.batch_size, shuffle=args.shuffle)

        if args.training_mode == 'two_classes':
            
            combined_dataset = ADdatasets.MergedDatasetWrapper([wrapper_inlier_complet, wrapper_anomaly_complet])
            combined_dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
            return {"complet": combined_dataloader}

        return {"inlier": inlier_dataloader, "anomaly": anomaly_dataloader}

    else:
        wrapper_inlier_train = ADdatasets.DatasetWrapper(inlier_dataset_train_emb, args.type_emb)
        wrapper_anomaly_train = ADdatasets.DatasetWrapper(anomaly_dataset_train_emb, args.type_emb)
        wrapper_inlier_test = ADdatasets.DatasetWrapper(inlier_dataset_test_emb, args.type_emb)
        wrapper_anomaly_test = ADdatasets.DatasetWrapper(anomaly_dataset_test_emb, args.type_emb)

        inlier_dataloader_train = DataLoader(wrapper_inlier_train, batch_size=args.batch_size, shuffle=args.shuffle)
        anomaly_dataloader_train = DataLoader(wrapper_anomaly_train, batch_size=args.batch_size, shuffle=args.shuffle)
        inlier_dataloader_test = DataLoader(wrapper_inlier_test, batch_size=args.batch_size, shuffle=args.shuffle)
        anomaly_dataloader_test = DataLoader(wrapper_anomaly_test, batch_size=args.batch_size, shuffle=args.shuffle)

        if args.training_mode == 'two_classes':
            combined_train_dataset = ADdatasets.MergedDatasetWrapper([wrapper_inlier_train, wrapper_anomaly_train])
            combined_test_dataset = ADdatasets.MergedDatasetWrapper([wrapper_inlier_test, wrapper_anomaly_test])

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

    start = time.time()

    logger.info(
        f"\nStarting execution with dataset='{args.dataset_name}', "
        f"training_mode='{args.training_mode}', "
        f"inlier_topic='{args.inlier_topic}', "
        f"type_tac='{args.type_tac}', "
        f"anomaly_rate={args.anomaly_rate}, "
        f"embedding='{args.type_emb}' ({args.model_name}). \n\n"
    )
    dl = data_preparation(args, logger, ADdatasets, tac, embedding_encoder)

    # training_mode = 'one_class' --> return train/test in any dataset there is anomaly and inlier subset
    # training_mode = 'two_classes' --> return train/test and separate anomaly and inlier subset to get 4 dataloaders
    if args.training_mode == 'two_classes':

        if args.full_dataset_ or args.dataset_name == 'WOS':
            dataloader_complet = dl['complet']
        else:
            dataloader_train = dl['train']
            dataloader_test = dl['test']

    else:
        if args.full_dataset_ or args.dataset_name == 'WOS':
            dataloader_train = dl['inlier']
            dataloader_test = dl['anomaly']
        else:
            inlier_dataloader_train = dl['inlier_train']
            anomaly_dataloader_train = dl['anomaly_train']

            inlier_dataloader_test = dl['inlier_test']
            anomaly_dataloader_test = dl['anomaly_test']

    end = time.time()
    logger.info(f"Data Preparation ends after : {end - start:.2f} seconds")

    if args.ad_model == 'ocsvm':

        clf, y_pred, scores = ocsvm.One_Class_SVM(dataloader_train.dataset.inputs, dataloader_train.dataset.labels, False)
        auc, f1, precision, recall, fpr95 = ev.evaluation(dataloader_train.dataset.labels, scores, y_pred, verbose=False)

        print(clf, end="\n\n")

        print(auc)
        print(f1)
        print(precision)
        print(recall)
        print(fpr95)

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
        "--model_name",
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



    args = parser.parse_args()
    main(args)

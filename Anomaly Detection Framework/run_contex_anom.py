import argparse
import logging
from Data_Preparation import utils
from Data_Preparation.Embedding import embedding_encoder
from Data_Preparation.Tac import tac


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):


    data_train, data_test = utils.import_dataset(name=args.dataset_name, batch_size=64)

    data_train = utils.preprocess(data_train.dataset)
    data_test = utils.preprocess(data_test.dataset)

    train_inlier, train_anomaly = tac.textual_anomaly_contamination(data_train, args.dataset_name, args.inlier_topic, args.type_tac, args.nu, True)

    print(train_inlier)

    print(train_anomaly)

























if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments script")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="20newsgroups"
    )

    parser.add_argument(
        "--inlier_topic",
        type=str,
        default="computer"
    )

    parser.add_argument(
        "--type_tac",
        type=str,
        default="ruff"
    )

    parser.add_argument(
        "--nu",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--type_encoder",
        type=str,
        default="sentencebert"
    )

    parser.add_argument(
        "--model_encoder",
        type=str,
        default="all-distilroberta-v1"
    )

    parser.add_argument(
        "--whichset",
        type=str,
        default="all"
    )

    parser.add_argument(
        "--nbruns",
        type=int,
        default=10
    )

    args = parser.parse_args()
    main(args)
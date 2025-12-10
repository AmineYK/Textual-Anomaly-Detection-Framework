import argparse
import logging
from Data_Preparation import utils
from Data_Preparation.Embedding import embedding_encoder
import torch
import os
from utils import create_tables

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

    # data_train, data_test = utils.import_dataset(name=args.dataset_name, batch_size=64)

    # data_train = utils.preprocess(data_train.dataset)
    # data_test = utils.preprocess(data_test.dataset)


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # sentencebertEncoder = embedding_encoder.EmbeddingEncoder(args.model_encoder, args.type_encoder, device)

    # save_dir = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Data"
    
    # inlier_topics = dataset_topics_dict[args.dataset_name]
    # print(inlier_topics)

    # for run in range(args.nbruns):

    #     for inlier_topic in inlier_topics:

    #         save_path_temp = os.path.join(save_dir, f"{args.dataset_name}/{inlier_topic}/run{run+1}")
    #         os.makedirs(save_path_temp, exist_ok=True)

    #         save_path = os.path.join(save_path_temp, f"ds_test_{inlier_topic}_run{run+1}.pt")
            
    #         if os.path.exists(save_path):
    #             print(f"{save_path} already done")
    #         else:
    #             _, _, _, X_test, y_test = utils.get_embeddings(sentencebertEncoder, data_train, data_test, inlier_topic, 
    #                                                      args.dataset_name, args.type_tac, args.nu, device, 'test', 'content')
    #             print(f"data saved at : {save_path}")
    #             torch.save({"X_test": X_test, "y_test": y_test}, save_path)

    create_tables()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Data Matrix script")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="20newsgroups"
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
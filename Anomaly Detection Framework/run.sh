#!/bin/bash

cd "$(dirname "$0")" || exit 1

# python3 main.py --dataset_name 20NewsGroups --inlier_topic science --type_tac ruff --anomaly_rate 0.1
# python3 main.py --dataset_name 20NewsGroups --inlier_topic science --type_tac pantin --anomaly_rate 0.1

# python3 main.py --dataset_name Reuters --inlier_topic energy --type_tac pantin --anomaly_rate 0.1
# python3 main.py --dataset_name Reuters --inlier_topic acq --type_tac ruff --anomaly_rate 0.1

# python3 main.py --dataset_name WOS --inlier_topic Civil_Engineering --type_tac pantin --anomaly_rate 0.1

# python3 main.py --dataset_name DBpedia14 --inlier_topic Animal --type_tac pantin --anomaly_rate 0.1

# python3 main.py --dataset_name Reuters --training_mode two_classes --inlier_topic energy --type_tac pantin --anomaly_rate 0.1 --model_name tfidf --type_emb tfidf --batch_size 32 --shuffle --ad_model ocsvm
# python3 main.py --dataset_name 20NewsGroups --training_mode two_classes --inlier_topic science --type_tac ruff --anomaly_rate 0.1 --model_name tfidf --type_emb tfidf --batch_size 32 --shuffle --ad_model ocsvm
# python3 main.py --dataset_name WOS --training_mode two_classes --inlier_topic Civil_Engineering --type_tac pantin --anomaly_rate 0.1 --model_name tfidf --type_emb tfidf --batch_size 32 --shuffle
# python3 main.py --dataset_name DBpedia14 --training_mode two_classes --inlier_topic Animal --type_tac pantin --anomaly_rate 0.1 --model_name tfidf --type_emb tfidf --batch_size 32 --shuffle
# python3 main.py --dataset_name AGNews --training_mode two_classes --inlier_topic Sports --type_tac fate --anomaly_rate 0.1 --model_name tfidf --type_emb tfidf --batch_size 32 --shuffle



# python3 main.py --dataset_name Reuters --training_mode two_classes --inlier_topic energy --type_tac pantin --anomaly_rate 0.1 --model_name glove_300d.kv --type_emb glove --batch_size 32 --shuffle


# python3 main.py --dataset_name AGNews --inlier_topic Sports --type_tac fate --anomaly_rate 0.1 --model_name tfidf --type_emb tfidf

# python3 main.py --dataset_name AGNews --inlier_topic Sports --type_tac fate --anomaly_rate 0.1 --model_name glove_300d.kv --type_emb glove

# python3 main.py --dataset_name AGNews --inlier_topic Sports --type_tac fate --anomaly_rate 0.1 --model_name fasttext_300d.kv --type_emb fasttext




python3 main.py --dataset_name 20NewsGroups --training_mode two_classes --inlier_topic science --type_tac ruff --anomaly_rate 0.1









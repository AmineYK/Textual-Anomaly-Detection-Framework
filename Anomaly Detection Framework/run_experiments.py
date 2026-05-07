import os
import torch
import numpy as np
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# =========================
# IMPORTS PROJET
# =========================
# Adapte ces imports selon ton arborescence
import sys
sys.path.append('./Textual-Anomaly-Detection-Framework/Anomaly Detection Framework')
from Modelisation.FlowMatching.flow_matching_transformers_token import FlowDiTToken, FlowMatchingTransformersToken
from Data_Preparation.utils import encode_tokens
from utils import load_data_inlier, load_data_test


# =========================
# MAIN
# =========================

def main():

    # -------------------------------------------------
    # DEVICE
    # -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------
    # PARAMS
    # -------------------------------------------------
    dataset_name = '20newsgroups'
    inlier_topic = 'computer'
    type_tac = None
    anomaly_rate = 0.1

    save_dir = (
        "/home/2017025/ayouce01/"
        "Textual-Anomaly-Detection-Framework/"
        "Anomaly Detection Framework/Data"
    )

    n_run = 2

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    print("\nLoading train data...")
    data_train = load_data_inlier(
        dataset_name,
        inlier_topic,
        save_dir,
        is_infec=False,
        is_cvdd=True
    )

    print("\nLoading test data...")
    data_test = load_data_test(
        dataset_name,
        inlier_topic,
        n_run,
        save_dir,
        is_cvdd=True
    )

    print("\nTrain data:")
    print(data_train)

    print("\nTest data:")
    print(data_test)

    # -------------------------------------------------
    # EMBEDDINGS
    # -------------------------------------------------
    X_inlier = Tensor(data_train['sbert_embeddings']).to(device)

    X_test = Tensor(
        data_test['sbert_embeddings']
    ).to(device)

    y_test = np.array(
        data_test['anomaly_class']
    )

    print(f"\nX_inlier shape: {X_inlier.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # -------------------------------------------------
    # TOKENIZER + BERT
    # -------------------------------------------------
    model_name = "roberta-base"
    # model_name = "microsoft/deberta-v3-base"

    print(f"\nLoading tokenizer/model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bertmodel = AutoModel.from_pretrained(
        model_name
    ).to(device)

    bertmodel.eval()

    # -------------------------------------------------
    # TEXT COLUMN
    # -------------------------------------------------
    # COL = 'content'
    COL = 'text'

    # -------------------------------------------------
    # ENCODE TRAIN TOKENS
    # -------------------------------------------------
    print("\nEncoding train tokens...")

    X_tokens_train, tokens_train, attentions_train_mask = encode_tokens(
        bertmodel,
        tokenizer,
        data_train[COL],
        device,
        batch_size=64,
        max_length=128,
        model_type='encoder'
    )

    # -------------------------------------------------
    # ENCODE TEST TOKENS
    # -------------------------------------------------
    print("\nEncoding test tokens...")

    X_tokens_test, tokens_test, attentions_test_mask = encode_tokens(
        bertmodel,
        tokenizer,
        data_test[COL],
        device,
        batch_size=64,
        max_length=128,
        model_type='encoder'
    )

    print(f"\nX_tokens_test shape: {X_tokens_test.shape}")

    # -------------------------------------------------
    # LOAD FLOW MODEL
    # -------------------------------------------------
    name = "computer_128_256_ep300"

    saving_path = (
        "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/"
        "Anomaly Detection Framework/"
        f"fm_trained_models/{name}"
    )

    print(f"\nLoading checkpoint: {saving_path}")

    checkpoint = torch.load(
        saving_path,
        map_location=device
    )

    config = checkpoint["config"]

    # -------------------------------------------------
    # BUILD MODEL
    # -------------------------------------------------
    flowmodel = FlowDiTToken(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        depth=config['depth'],
        n_heads=config['n_heads']
    ).to(device)

    flowmodel.load_state_dict(
        checkpoint["model"]
    )

    flowmodel.eval()

    # -------------------------------------------------
    # FLOW MATCHING WRAPPER
    # -------------------------------------------------
    fm_transformer = FlowMatchingTransformersToken(
        flowmodel,
        config['source'],
        config['target'],
        config,
        noise_is_target=True,
        rectified=None
    )

    # -------------------------------------------------
    # EULER INTEGRATION
    # -------------------------------------------------
    print("\nRunning Euler integration...")

    with torch.no_grad():

        x_final, velocities, x_inter = (
            fm_transformer.euler_integrate(
                X_tokens_test,
                attentions_test_mask,
                5,
                True
            )
        )

    print("\nIntegration done.")

    # -------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------
    output_dir = (
        "./saved_fm_outputs/"
        f"{dataset_name}_{inlier_topic}_run{n_run}"
    )

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving outputs in: {output_dir}")

    torch.save(
        x_final.cpu(),
        os.path.join(output_dir, "x_final.pt")
    )

    torch.save(
        velocities.cpu(),
        os.path.join(output_dir, "velocities.pt")
    )

    torch.save(
        x_inter.cpu(),
        os.path.join(output_dir, "x_inter.pt")
    )

    print("\nSaved files:")
    print(f"- {os.path.join(output_dir, 'x_final.pt')}")
    print(f"- {os.path.join(output_dir, 'velocities.pt')}")
    print(f"- {os.path.join(output_dir, 'x_inter.pt')}")

    print("\nDone.")


# =========================
# ENTRYPOINT
# =========================

if __name__ == "__main__":
    main()
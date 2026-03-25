import argparse
import logging
from Data_Preparation import utils
from Data_Preparation.Embedding import embedding_encoder
from Data_Preparation.Tac import tac
import torch
import numpy as np
from torch import Tensor
import os
from utils import load_data_inlier, load_data_test, load_hyperparams, get_data_fasttext
from Modelisation.FlowMatching.flow_matching_transformers import *
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def trajectory_direction(fmt, X, N_steps=20, device=device):
        """
        Retourne la distance au centroid à chaque timestep
        et la direction (convergente ou divergente)
        """
        fmt.model.eval()
        with torch.no_grad():
            trajectory = [X.clone()]
            distances  = [torch.norm(X - fmt.centroid, dim=1)]  # (B,)
            
            dt = 1.0 / N_steps
            x_t = X.clone()
            
            for i in range(N_steps):
                t_val = i * dt
                t = torch.full((x_t.shape[0],), t_val, device=device)
                v = fmt.model(x_t, t)
                x_t = x_t + dt * v
                
                dist = torch.norm(x_t - fmt.centroid, dim=1)  # (B,)
                trajectory.append(x_t.clone())
                distances.append(dist)
            
            distances = torch.stack(distances)  # (N_steps+1, B)
            
            # Dérivée discrète : dist[t+1] - dist[t]
            delta_dist = distances[1:] - distances[:-1]  # (N_steps, B)
            
            # Direction globale : convergente ou divergente ?
            # On regarde si la distance finale < distance initiale
            global_direction = distances[-1] - distances[0]  # (B,)
            # négatif = convergente, positif = divergente
            
        return distances.cpu(), delta_dist.cpu(), global_direction.cpu()


    dataset_name = 'sms'
    inlier_topic = 'normal'
    type_tac = None
    anomaly_rate = 0.1
    save_dir = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Data"
    n_run = 6
    name = f"{dataset_name}_{inlier_topic}_{n_run}_push1e-4"
    saving_path = f"/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/fm_trained_models/{name}"

    data_train = load_data_inlier(dataset_name, inlier_topic, save_dir, is_infec=False, is_cvdd=True)
    data_test = load_data_test(dataset_name, inlier_topic, n_run, save_dir, is_cvdd=True)

    print(data_train)
    print(data_test)

    X_inlier = Tensor(data_train['sbert_embeddings']).to(device)[:5000]
    X_test =  Tensor(data_test['sbert_embeddings']).to(device)
    y_test = np.array(data_test['anomaly_class'])

    config = {
            'latent_dim': 768,
            'hidden_dim': 256,
            'depth': 8,
            'n_heads': 8,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'lambda_svdd': 1e-3,
            'lambda_push': 1e-3,
            'lambda_margin': 1e-4,
            'epochs': 500,
            'lr_epochs': 100,
            'warmup_epochs': 0,
            'grad_clip': 0.5,
            'flow_type': 'linear',  
            'sigma': 0.1, 
            'batch_size' : 126,
            'lambda_reg_angle': None,
            'lambda_reg_kl': None,
            'n_step_euler_integrate' : 1,
            'coef_var': 1,
            'rate_neg_batch':1.0,
            'sig_levels_neg' : [0.3, 0.5],
            'target' : 'gaussian-neigh',
            'source' : X_inlier.to(device)
    }
    noise_is_target = True
    naming_dis = {
        'target' : config['target'] if noise_is_target else 'SBert', 
        'source' : config['source'] if not noise_is_target else 'SBert'
    }

    for _ in range(1):

        print("LOSS PUSH DIST : NEGATIVE DIM PERTUB")

        flowmodel = FlowDiT(
                    latent_dim=config['latent_dim'],
                    hidden_dim=config['hidden_dim'],
                    depth=config['depth'],
                    n_heads=config['n_heads']
            ).to(device)

        fm_transformer = FlowMatchingTransformers(flowmodel, config['source'], config['target'], config, noise_is_target=noise_is_target, rectified=None)
        total_loss_liste, loss_fm_liste, loss_svdd_liste, loss_push_liste, margin_value_liste, r_in_value_liste = fm_transformer.train(True)

        print(fm_transformer.test(X_test, y_test, X_inlier, 'norm-centroid'))

        epochs = np.arange(len(total_loss_liste))

        r_in = np.array(r_in_value_liste)
        margin = np.array(margin_value_liste)
        r_out = r_in + margin

        fig, axs = plt.subplots(1, 5, figsize=(20, 5))

        axs[0].plot(epochs, total_loss_liste, label="total_loss")
        axs[0].set_title("Total Loss")
        axs[0].legend()

        axs[1].plot(epochs, loss_fm_liste, label="loss_fm")
        axs[1].set_title("Loss FM")
        axs[1].legend()

        axs[2].plot(epochs, loss_svdd_liste, label="loss_svdd")
        axs[2].set_title("Loss SVDD")
        axs[2].legend()

        axs[3].plot(epochs, loss_push_liste, label="loss_push")
        axs[3].set_title("Losses Push")
        axs[3].legend()

        axs[4].plot(epochs, r_in, label="r_in")
        # axs[3].plot(epochs, margin, label="margin")
        axs[4].plot(epochs, r_out, label="r_out")
        axs[4].set_title("Radii Evolution")
        axs[4].legend()

        plt.tight_layout()
        plt.show()

        # --------------------------
        # Train Inliers
        # --------------------------
        dist_train_inlier = torch.sum(
            (fm_transformer.forward_flow(X_inlier, 'midpoint', 8)[-1] - fm_transformer.centroid) ** 2,
            dim=1
        )

        rate_train_inlier_r_in = ((dist_train_inlier < fm_transformer.r_in).float().mean() * 100).item()
        rate_train_inlier_r_in_r_out = (((dist_train_inlier >= fm_transformer.r_in) & (dist_train_inlier < fm_transformer.r_out)).float().mean() * 100).item()
        rate_train_inlier_r_out = ((dist_train_inlier >= fm_transformer.r_out).float().mean() * 100).item()

        print(f"% Train-inliers dans r_in : {rate_train_inlier_r_in:.1f}%")
        print(f"% Train-inliers entre r_in et r_out : {rate_train_inlier_r_in_r_out:.1f}%")
        print(f"% Train-inliers dans r_out : {rate_train_inlier_r_out:.1f}%")
        print("--------------------------")

        nb_samples_neg = 130

        sigma_levels = [
            # 0.3 * torch.sqrt(fm_transformer.var),
            # 0.5 * torch.sqrt(fm_transformer.var)
            config['sig_levels_neg'][0] * torch.sqrt(fm_transformer.var),
            config['sig_levels_neg'][1] * torch.sqrt(fm_transformer.var)
        ]
        x_0_negative = []

        for i,sig in enumerate(sigma_levels):

            eps = sig * torch.randn((i+1)*nb_samples_neg, 768).to(device)
            x_0_negative.extend(fm_transformer.centroid + eps)

        x_0_negative = torch.stack(x_0_negative).to(device)

        # --------------------------
        # Train Pseudo-Anomalies
        # --------------------------
        dist_train_pseudo = torch.sum(
            (fm_transformer.forward_flow(x_0_negative, 'midpoint', 8)[-1] - fm_transformer.centroid) ** 2,
            dim=1
        )

        rate_train_pseudo_r_in = ((dist_train_pseudo < fm_transformer.r_in).float().mean() * 100).item()
        rate_train_pseudo_r_in_r_out = (((dist_train_pseudo >= fm_transformer.r_in) & (dist_train_pseudo < fm_transformer.r_out)).float().mean() * 100).item()
        rate_train_pseudo_r_out = ((dist_train_pseudo >= fm_transformer.r_out).float().mean() * 100).item()

        print(f"% Train-PseudoAnom dans r_in : {rate_train_pseudo_r_in:.1f}%")
        print(f"% Train-PseudoAnom entre r_in et r_out : {rate_train_pseudo_r_in_r_out:.1f}%")
        print(f"% Train-PseudoAnom dans r_out : {rate_train_pseudo_r_out:.1f}%")
        print("--------------------------")

        # --------------------------
        # Test Inliers
        # --------------------------
        X_test_inlier = X_test[y_test == 0]

        dist_test_inlier = torch.sum(
            (fm_transformer.forward_flow(X_test_inlier, 'midpoint', 8)[-1] - fm_transformer.centroid) ** 2,
            dim=1
        )

        rate_test_inlier_r_in = ((dist_test_inlier < fm_transformer.r_in).float().mean() * 100).item()
        rate_test_inlier_r_in_r_out = (((dist_test_inlier >= fm_transformer.r_in) & (dist_test_inlier < fm_transformer.r_out)).float().mean() * 100).item()
        rate_test_inlier_r_out = ((dist_test_inlier >= fm_transformer.r_out).float().mean() * 100).item()

        print(f"% Test-inliers dans r_in : {rate_test_inlier_r_in:.1f}%")
        print(f"% Test-inliers entre r_in et r_out : {rate_test_inlier_r_in_r_out:.1f}%")
        print(f"% Test-inliers dans r_out : {rate_test_inlier_r_out:.1f}%")
        print("--------------------------")

        # --------------------------
        # Test Anomalies
        # --------------------------
        X_test_anom = X_test[y_test == 1]

        dist_test_anom = torch.sum(
            (fm_transformer.forward_flow(X_test_anom, 'midpoint', 8)[-1] - fm_transformer.centroid) ** 2,
            dim=1
        )

        rate_test_anom_r_in = ((dist_test_anom < fm_transformer.r_in).float().mean() * 100).item()
        rate_test_anom_r_in_r_out = (((dist_test_anom >= fm_transformer.r_in) & (dist_test_anom < fm_transformer.r_out)).float().mean() * 100).item()
        rate_test_anom_r_out = ((dist_test_anom >= fm_transformer.r_out).float().mean() * 100).item()

        print(f"% Test-anomalies dans r_in : {rate_test_anom_r_in:.1f}%")
        print(f"% Test-anomalies entre r_in et r_out : {rate_test_anom_r_in_r_out:.1f}%")
        print(f"% Test-anomalies dans r_out : {rate_test_anom_r_out:.1f}%")
        print("--------------------------")

        # --------------------------
        # Statistiques globales
        # --------------------------
        print(f"dist moyenne Train inliers     : {dist_train_inlier.mean():.4f}")
        print(f"dist moyenne Train pseudoAno   : {dist_train_pseudo.mean():.4f}")
        print(f"dist moyenne Test inliers      : {dist_test_inlier.mean():.4f}")
        print(f"dist moyenne Test anomalies    : {dist_test_anom.mean():.4f}")
        print(f"Gap Test : inliers/anomalies   : {dist_test_anom.mean() - dist_test_inlier.mean():.4f}")
        d_cohen = (dist_test_anom.mean() - dist_test_inlier.mean()) / (
        torch.sqrt((dist_test_inlier.var() + dist_test_anom.var()) / 2)
        )
        print(f"Cohen's d : {d_cohen:.4f}") 
        print(f"r_in                            : {fm_transformer.r_in.item():.4f}")
        print(f"r_out                           : {fm_transformer.r_out.item():.4f}")
        print()
        distances, delta_dist, global_direction = trajectory_direction(fm_transformer, X_test, N_steps=50, device=device) 
        print(f" Test Nb INLIER Examples diverge from centroid {(global_direction[y_test == 0][global_direction[y_test == 0] > 0].shape[0] / X_test.shape[0])*100}%") 
        print(f" Test Nb INLIER Examples converge to centroid {(global_direction[y_test == 0][global_direction[y_test == 0] < 0].shape[0] / X_test.shape[0])*100}%")  

        print(f" Test Nb ANOMALIES Examples diverge from centroid {(global_direction[y_test == 1][global_direction[y_test == 1] > 0].shape[0] / X_test.shape[0])*100}%") 
        print(f" Test Nb ANOMALIES Examples converge to centroid {(global_direction[y_test == 1][global_direction[y_test == 1] < 0].shape[0] / X_test.shape[0])*100}%")  
        print()    
        distances, delta_dist, global_direction = trajectory_direction(fm_transformer, X_inlier, N_steps=50, device=device)   
        print(f" Train Nb Examples diverge from centroid {(global_direction[global_direction > 0].shape[0] / X_inlier.shape[0])*100}%") 
        print(f" Train Nb Examples converge to centroid {(global_direction[global_direction < 0].shape[0] / X_inlier.shape[0])*100}%")   



        # checkpoint = {
        #     "model": flowmodel.state_dict(),
        #     "config": config
        # }

        # torch.save(checkpoint, saving_path)






























    # save_dir = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Data"
    # names = ['enron', 'mage']
    # inlier_topics = ['normal']


    # for name in names:
    #     print(name)
    #     for inlier_topic in inlier_topics:
    #         print(inlier_topic)
    #         data_train = load_data_inlier(name, inlier_topic, save_dir, is_infec=False, is_cvdd=True)
    #         X_inlier = Tensor(data_train['sbert_embeddings']).to(device)
    #         print(X_inlier.shape)

    #         for n_run in range(1, 11):
    #             data_test = load_data_test(name, inlier_topic, n_run, save_dir, is_cvdd=True)
    #             y_test = Tensor(data_test['anomaly_class'])
    #             X_test = Tensor(data_test['sbert_embeddings']).to(device)
    #             if n_run == 4:
    #                 print(y_test.shape, X_test.shape)
    #                 print(np.unique(y_test, return_counts=True))
    #                 print("-------------------------")

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
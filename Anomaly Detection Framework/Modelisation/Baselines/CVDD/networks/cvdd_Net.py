import torch
import torch.nn as nn
import torch.nn.functional as F

from .self_attention import SelfAttention
from .embedding_layer import BERTEmbeddingEncoder
from .utils import initialize_context_vectors
import numpy as np
import torch.optim as optim
import time
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
from Modelisation.evaluation import fpr95_score


class CVDDNet(nn.Module):

    def __init__(self, pretrained_model, attention_size=100, n_attention_heads=1):
        super().__init__()
        # Load pretrained model (which provides a hidden representation per word, e.g. word vector or language model)
        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.embedding_size

        # Set self-attention module
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads
        self.self_attention = SelfAttention(hidden_size=self.hidden_size,
                                            attention_size=attention_size,
                                            n_attention_heads=n_attention_heads)

        # Model parameters
        self.c = nn.Parameter((torch.rand(1, n_attention_heads, self.hidden_size) - 0.5) * 2)
        self.cosine_sim = nn.CosineSimilarity(dim=2)

        # Temperature parameter alpha
        self.alpha = 0.0

    def forward(self, x):
        # x.shape = (sentence_length, batch_size)

        hidden = self.pretrained_model(x)
        if isinstance(self.pretrained_model, BERTEmbeddingEncoder):
            hidden = hidden.permute(1,0,2)
            
        # hidden.shape = (sentence_length, batch_size, hidden_size)

        M, A = self.self_attention(hidden)
        # A.shape = (batch_size, n_attention_heads, sentence_length)
        # M.shape = (batch_size, n_attention_heads, hidden_size)

        cosine_dists = 0.5 * (1 - self.cosine_sim(M, self.c))
        context_weights = F.softmax(-self.alpha * cosine_dists, dim=1)

        return cosine_dists, context_weights, A
    

class CVDDTrainer:

    def __init__(self, optimizer_name='adam', learning_rate=1e-3, lr_milestones=(20,70), n_epochs=100, 
                 lambda_p=0.0, alpha_scheduler='hard', weight_decay=1e-6):
        
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.optimizer = None
        self.scheduler = None
        
        self.lr_milestones = lr_milestones
        self.n_epochs = n_epochs
        self.lambda_p = lambda_p
        self.alpha_scheduler = alpha_scheduler
        
        self.train_dists = None
        self.train_att_matrix = None
        self.train_top_words = None
        self.c = None
        self.training_time = None

        self.test_dists = None
        self.test_att_matrix = None
        self.test_top_words = None
        self.test_auc = 0.0
        self.test_scores = None
        self.test_att_weights = None


    def train(self, model, dl_train):

        logger = logging.getLogger()
        logger.info('Starting training...')
        start_time = time.time()

        model.c.data = torch.from_numpy(
                initialize_context_vectors(model, dl_train)[np.newaxis, :])
        parameters = filter(lambda p: p.requires_grad, model.parameters())

        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(parameters, lr=self.learning_rate)
        else : raise Exception("Please choose an optimizer in this list ('adam','sgd')")

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_milestones, gamma=0.1)

        alpha_milestones = np.arange(1, 6) * int(self.n_epochs / 5)  # 5 equidistant milestones over n_epochs
        if self.alpha_scheduler == 'soft':
            alphas = [0.0] * 5
        if self.alpha_scheduler == 'linear':
            alphas = np.linspace(.2, 1, 5)
        if self.alpha_scheduler == 'logarithmic':
            alphas = np.logspace(-4, 0, 5)
        if self.alpha_scheduler == 'hard':
            alphas = [100.0] * 4

        model.train()
        alpha_i = 0

        for epoch in range(self.n_epochs):
            
            self.scheduler.step()

            if epoch in self.lr_milestones:
                logger.info(f"LR scheduler: new learning rate is %g" % float(self.scheduler.get_last_lr()[0]))

            if epoch in alpha_milestones:
                model.alpha = float(alphas[alpha_i])
                logger.info('  Temperature alpha scheduler: new alpha is %g' % model.alpha)
                alpha_i += 1

            epoch_loss = 0.0
            n_batches = 0
            att_matrix = np.zeros((model.n_attention_heads, model.n_attention_heads))
            dists_per_head = ()
            epoch_start_time = time.time()

            for inputs, labels, texts, idx in dl_train:

                inputs = inputs.transpose(0, 1)
                
                self.optimizer.zero_grad() 

                cosine_dists, context_weights, A = model(inputs)
                scores = context_weights * cosine_dists

                I = torch.eye(model.n_attention_heads)
                CCT = model.c @ model.c.transpose(1, 2)
                P = torch.mean((CCT.squeeze() - I) ** 2)

                loss_P = self.lambda_p * P
                loss_emp = torch.mean(torch.sum(scores, dim=1))
                loss = loss_emp + loss_P

                dists_per_head += (cosine_dists.cpu().data.numpy(),)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # clip gradient norms in [-0.5, 0.5]
                self.optimizer.step()

                AAT = A @ A.transpose(1, 2)
                att_matrix += torch.mean(AAT, 0).cpu().data.numpy()

                epoch_loss += loss.item()
                n_batches += 1


            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
            f'| Train Loss: {epoch_loss / n_batches:.6f} |')


            self.train_dists = np.concatenate(dists_per_head)
            self.train_att_matrix = att_matrix / n_batches
            self.train_att_matrix = self.train_att_matrix.tolist()


        c = np.squeeze(model.c.data.numpy())
        c = c.tolist()

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training. \n')
        
        return model

    from sklearn.metrics import roc_auc_score, average_precision_score

    def test(self, model, dl_test, ad_score='context_dist_mean'):

        logger = logging.getLogger()
        logger.info('\nStarting testing...')

        n_attention_heads = model.n_attention_heads
        epoch_loss = 0.0
        n_batches = 0
        att_matrix = np.zeros((n_attention_heads, n_attention_heads))
        dists_per_head = ()
        idx_label_score_head = []
        att_weights = []
        start_time = time.time()
        model.eval()

        with torch.no_grad():
            for inputs, labels, texts, idx in dl_test:

                cosine_dists, context_weights, A = model(inputs)
                scores = context_weights * cosine_dists
                _, best_att_head = torch.min(scores, dim=1)

                I = torch.eye(n_attention_heads)
                CCT = model.c @ model.c.transpose(1, 2)
                P = torch.mean((CCT.squeeze() - I) ** 2)

                loss_P = self.lambda_p * P
                loss_emp = torch.mean(torch.sum(scores, dim=1))
                loss = loss_emp + loss_P

                dists_per_head += (cosine_dists.cpu().data.numpy(),)
                ad_scores = torch.mean(cosine_dists, dim=1)

                idx_label_score_head += list(zip(
                    idx,
                    labels.cpu().data.numpy().tolist(),
                    ad_scores.cpu().data.numpy().tolist(),
                    best_att_head.cpu().data.numpy().tolist()
                ))

                att_weights += A[best_att_head][:][range(len(idx))].cpu().data.numpy().tolist()

                AAT = A @ A.transpose(1, 2)
                att_matrix += torch.mean(AAT, 0).cpu().data.numpy()

                epoch_loss += loss.item()
                n_batches += 1

        test_dists = np.concatenate(dists_per_head)
        test_att_matrix = att_matrix / n_batches
        test_att_matrix = test_att_matrix.tolist()

        self.test_scores = idx_label_score_head
        self.test_att_weights = att_weights

        # Extract labels and scores
        _, labels, scores, _ = zip(*idx_label_score_head)
        labels = np.array(labels)
        scores = np.array(scores)

        # Compute metrics
        if np.sum(labels) > 0:
            best_context = None
            if ad_score == 'context_dist_mean':
                test_auc = roc_auc_score(labels, scores)
                test_ap = average_precision_score(labels, scores)
                test_fpr95 = fpr95_score(labels, scores)
            elif ad_score == 'context_best':
                test_auc = 0.0
                test_ap = 0.0
                test_fpr95 = 0.0
                for context in range(n_attention_heads):
                    auc_candidate = roc_auc_score(labels, test_dists[:, context])
                    if auc_candidate > test_auc:
                        test_auc = auc_candidate
                        best_context = context
                        test_ap = average_precision_score(labels, test_dists[:, context])
                        test_fpr95 = fpr95_score(labels, test_dists[:, context])
        else:
            best_context = None
            test_auc = test_ap = test_fpr95 = 0.0

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.4f}'.format(test_auc))
        logger.info('Test AP: {:.4f}'.format(test_ap))
        logger.info('Test FPR95: {:.4f}'.format(test_fpr95))
        logger.info(f'Test Best Context: {best_context}')
        logger.info('Finished testing.')

        return test_auc, test_ap, test_fpr95, best_context











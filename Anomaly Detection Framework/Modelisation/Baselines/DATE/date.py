import torch
import torch.nn as nn

from Modelisation.Baselines.DATE.utils import *
from Modelisation.Baselines.baseline import BaselineModel
from torch.utils.data import Dataset, DataLoader
import Modelisation.evaluation as ev

from transformers import (
    BertConfig, BertForMaskedLM, BertModel, BertTokenizerFast,
    ElectraConfig, ElectraForMaskedLM, ElectraModel, ElectraTokenizerFast,
    AlbertConfig, AlbertForMaskedLM, AlbertModel, AlbertTokenizer
)

class DateGenerator(nn.Module):
    def __init__(self, which_config, vocab_size):
        super().__init__()

        if which_config == "bert":
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                hidden_act="gelu",
                max_position_embeddings=512
            )
            self.model = BertForMaskedLM(config)
            
        elif which_config == "electra":
            config = ElectraConfig(
                vocab_size=vocab_size,
                embedding_size=128,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                hidden_act="gelu",
                max_position_embeddings=512
            )
            self.model = ElectraForMaskedLM(config)
            
        elif which_config == "albert":
            config = AlbertConfig(
                vocab_size=vocab_size,
                embedding_size=128,  
                hidden_size=256,
                num_hidden_layers=4,
                num_hidden_groups=1,  
                num_attention_heads=4,
                intermediate_size=1024,
                hidden_act="gelu_new",  
                max_position_embeddings=512
            )
            self.model = AlbertForMaskedLM(config)
        else:
            raise ValueError(f"Unknown config: {which_config}")

    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return out.logits


class DateDiscriminator(nn.Module):
    def __init__(self, which_config, vocab_size, K):
        super().__init__()

        if which_config == "bert":
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                hidden_act="gelu",
                max_position_embeddings=512
            )
            self.encoder = BertModel(config)
            
        elif which_config == "electra":
            config = ElectraConfig(
                vocab_size=vocab_size,
                embedding_size=128,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                hidden_act="gelu",
                max_position_embeddings=512
            )
            self.encoder = ElectraModel(config)
            
        elif which_config == "albert":
            config = AlbertConfig(
                vocab_size=vocab_size,
                embedding_size=128,
                hidden_size=256,
                num_hidden_layers=4,
                num_hidden_groups=1,  
                num_attention_heads=4,
                intermediate_size=1024,
                hidden_act="gelu_new",  
                max_position_embeddings=512
            )
            self.encoder = AlbertModel(config)
        else:
            raise ValueError(f"Unknown config: {which_config}")

        # RTD head: binary classification per token
        self.rtd_head = nn.Linear(config.hidden_size, 1)

        # RMD head: K-way classification (CLS → K masks)
        self.rmd_head = nn.Linear(config.hidden_size, K)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = out.last_hidden_state  # [batch, seq_len, hidden_size]
        cls = hidden[:, 0]  # [batch, hidden_size]

        rtd_logits = self.rtd_head(hidden).squeeze(-1)  # [batch, seq_len]
        rmd_logits = self.rmd_head(cls)  # [batch, K]

        return rtd_logits, rmd_logits
    


class DATEDataset(Dataset):

    def __init__(self, texts, labels=None, tokenizer=None, max_len=498):
        self.texts = texts
        self.labels = labels if labels is not None else [0] * len(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            self.encodings["input_ids"][idx],
            self.encodings["attention_mask"][idx],
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

class DATEModel(BaselineModel):
    def __init__(self, args):

        self.which_config = args['which_config']
        self.encoder_name = args['encoder_name']

        if self.which_config =='albert':
            tokenizerObject = AlbertTokenizer
        elif self.which_config =='bert': 
            tokenizerObject = BertTokenizerFast
        else: tokenizerObject = ElectraTokenizerFast
        
        
        self.tokenizer =  tokenizerObject.from_pretrained(self.encoder_name)
        self.mask_token_id = self.tokenizer.mask_token_id

        self.device = args['device']
        self.K = args['K']
        # self.vocab_size = self.tokenizer.vocab_size
        self.vocab_size = len(self.tokenizer)

        self.generator = DateGenerator(self.which_config, self.vocab_size).to(self.device)
        self.discriminator = DateDiscriminator(self.which_config, self.vocab_size, self.K).to(self.device)

        self.lr = args['lr']
        self.weight_decay = args['weight_decay']

        self.optimizer = torch.optim.AdamW(
            list(self.generator.parameters()) + list(self.discriminator.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=True
        )

        self.seq_len = args['seq_len']
        self.ratio = args ['ratio']

        self.mask_patterns = generate_mask_patterns(
            K=self.K,
            seq_len=self.seq_len,
            ratio=self.ratio
        ).to(self.device)

        self.n_epochs = args['n_epochs']
        self.batch_size = args['batch_size']

    def train(self, data_train):
        """
        VERSION FINALE CORRIGÉE avec :
        1. Random generator (meilleur que paramétré)
        2. Masking dynamique par séquence
        3. Logging détaillé pour debug
        """
        train_texts = data_train['text']
        train_ds = DATEDataset(train_texts, None, self.tokenizer, max_len=self.seq_len)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self.discriminator.train()
        
        print(f"Training sur {len(train_texts)} samples, {len(train_loader)} batches")
        print(f"K={self.K} masks, ratio={self.ratio}, seq_len={self.seq_len}")

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            total_rmd = 0.0
            total_rtd = 0.0
            n_masked_tokens = 0

            for step, (input_ids, attention_mask, _) in enumerate(train_loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                # Sample UN mask pattern pour tout le batch
                k = torch.randint(0, self.K, (1,)).item()
                mask = self.mask_patterns[k]  # [seq_len]

                # Apply masking avec correction pour tokens spéciaux
                masked_ids, actual_mask = apply_mask_safe(input_ids, mask, self.tokenizer)
                
                # Compter combien de tokens réellement masqués
                n_masked_tokens += actual_mask.sum().item()

                # Random generator : remplacer [MASK] par tokens aléatoires
                corrupted_ids = corrupt_input_random(
                    masked_ids, 
                    self.tokenizer.mask_token_id,
                    self.vocab_size
                )

                # RTD labels : 1 si token remplacé, 0 sinon
                rtd_labels = (corrupted_ids != input_ids).long()
                
                # Exclure les tokens spéciaux de la loss RTD
                rtd_labels[input_ids == self.tokenizer.cls_token_id] = -100
                rtd_labels[input_ids == self.tokenizer.pad_token_id] = -100
                rtd_labels[attention_mask == 0] = -100

                # RMD labels : quel mask pattern a été utilisé
                rmd_labels = torch.full(
                    (input_ids.size(0),),
                    k,
                    device=self.device,
                    dtype=torch.long
                )

                # Forward discriminator
                rtd_logits, rmd_logits = self.discriminator(
                    corrupted_ids, attention_mask
                )

                # Loss computation
                ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
                bce = torch.nn.BCEWithLogitsLoss(reduction="none")
                
                # RTD loss (token-level)
                valid = rtd_labels != -100
                if valid.sum() > 0:
                    loss_rtd = bce(
                        rtd_logits[valid].float(),
                        rtd_labels[valid].float()
                    ).mean()
                else:
                    loss_rtd = torch.tensor(0.0, device=self.device)
                
                # RMD loss (sequence-level)
                loss_rmd = ce(rmd_logits, rmd_labels)
                
                # Combined loss
                mu = 100.0
                lambda_rtd = 50.0
                loss = mu * loss_rmd + lambda_rtd * loss_rtd

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), max_norm=1.0
                )
                
                self.optimizer.step()

                total_loss += loss.item()
                total_rmd += loss_rmd.item()
                total_rtd += loss_rtd.item()

            # Epoch stats
            avg_loss = total_loss / len(train_loader)
            avg_rmd = total_rmd / len(train_loader)
            avg_rtd = total_rtd / len(train_loader)
            avg_masked = n_masked_tokens / (len(train_loader) * self.batch_size * self.seq_len)
            
            if epoch % max(1, self.n_epochs // 5) == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs} - Loss: {avg_loss:.2f} "
                      f"(RMD: {avg_rmd:.4f}, RTD: {avg_rtd:.4f}) "
                      f"| Masked: {avg_masked:.2%}")

    def test(self, data_test):
        """
        CORRECTION : Ne pas inverser le score !
        date_anomaly_score retourne déjà un score où HIGHER = more normal
        """
        test_texts = data_test['text']
        test_labels = data_test['anomaly_class']        
        
        test_ds = DATEDataset(test_texts, test_labels, self.tokenizer)
        test_loader = DataLoader(test_ds, batch_size=64)

        scores, _ = date_anomaly_score(
            self.discriminator, test_loader, self.tokenizer, self.device
        )
    
        # CORRECTION : Ne PAS inverser !
        # scores élevés = normal, scores faibles = anomalie
        # Pour AUROC, on veut un score où anomalie = valeur élevée
        # Donc on INVERSE le score
        test_scores = -scores  # ou 1 - scores

        auc, ap, fpr95 = ev.evaluation(test_labels, test_scores, verbose=False)

        return auc, fpr95, ap

    # def train(self, data_train):

    #     train_texts = data_train['text']
    #     train_ds = DATEDataset(train_texts, None, self.tokenizer)
    #     train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

    #     self.generator.train()
    #     self.discriminator.train()

    #     for epoch in range(self.n_epochs):
    #         total_loss = 0.0

    #         for step, (input_ids, attention_mask, _) in enumerate(train_loader):
    #             input_ids = input_ids.to(self.device)
    #             attention_mask = attention_mask.to(self.device)

    #             # Sample a mask pattern
    #             k = torch.randint(0, self.mask_patterns.size(0), (1,)).item()
    #             mask = self.mask_patterns[k].to(self.device)

    #             # Apply masking
    #             masked_ids = apply_mask_safe(input_ids, mask, self.tokenizer)

    #             # MLM labels
    #             mlm_labels = input_ids.clone()
    #             mlm_labels[masked_ids != self.tokenizer.mask_token_id] = -100

    #             # Generator forward (MLM)
    #             mlm_logits = self.generator(masked_ids, attention_mask)

    #             # Corrupt input
    #             corrupted_ids = corrupt_input(
    #                 self.generator, masked_ids, attention_mask, self.tokenizer.mask_token_id
    #             )

    #             # Build RTD labels (CORRECTED)
    #             rtd_labels = (corrupted_ids != input_ids).long()
    #             rtd_labels[input_ids == self.tokenizer.cls_token_id] = -100
    #             rtd_labels[input_ids == self.tokenizer.sep_token_id] = -100
    #             rtd_labels[input_ids == self.tokenizer.pad_token_id] = -100
    #             rtd_labels[attention_mask == 0] = -100

    #             # RMD labels
    #             rmd_labels = torch.full(
    #                 (input_ids.size(0),),
    #                 k,
    #                 device=self.device
    #             )

    #             # Discriminator forward
    #             rtd_logits, rmd_logits = self.discriminator(
    #                 corrupted_ids, attention_mask
    #             )

    #             # Compute loss
    #             loss = date_loss(
    #                 rtd_logits, rmd_logits,
    #                 rtd_labels, rmd_labels,
    #                 mlm_logits, mlm_labels,
    #                 mu=100.0, lambda_rtd=50.0
    #             )

    #             # Backward
    #             loss.backward()
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()

    #             total_loss += loss.item()

    #         avg_loss = total_loss / len(train_loader)
    #         if epoch % (self.n_epochs // 3 ) == 0 :
    #             print(f"Epoch {epoch+1}/{self.n_epochs} - Loss: {avg_loss:.4f}")

    # def test(self, data_test):

    #     test_texts = data_test['text']
    #     test_labels = data_test['anomaly_class']        
        
    #     test_ds = DATEDataset(test_texts, test_labels, self.tokenizer)
    #     test_loader = DataLoader(test_ds, batch_size=64)

    #     scores, _ = date_anomaly_score(
    #         self.discriminator, test_loader, self.tokenizer, self.device
    #     )
    
    #     # Higher score = more normal, so we invert for anomaly detection
    #     test_scores = 1 - scores

    #     auc, ap, fpr95 = ev.evaluation(test_labels, test_scores, verbose=False)

    #     return auc, fpr95, ap
    
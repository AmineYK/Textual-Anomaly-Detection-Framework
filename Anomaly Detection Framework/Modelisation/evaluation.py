import numpy as np 
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score



def fpr95_score(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)  # 1 = anomalie
    idx = np.where(tpr >= 0.95)[0][0]
    return fpr[idx]


# Cosinus Similarity
#--------------------------------

def cosinus_similarity(emb1,emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def evaluation(y_true, scores, y_pred, verbose=True):
    
    auc = roc_auc_score(y_true, scores)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # FPR@95 = proportion d’anomalies mal classées comme normales quand 95 % des données normales sont correctement détectées.
    fpr95 = fpr95_score(y_true, scores)

    if verbose:
        print(f"AUC:        {auc:.4f}")
        print(f"F1-score:   {f1:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"Fpr95:     {fpr95:.4f}")
        

    return auc, f1, precision, recall, fpr95

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pyod.models.ocsvm import OCSVM
import time

# X = np.vstack([np.array(p_inlier_dl_reuters_glove.dataset['glove_embedding']), np.array(p_anomaly_dl_reuters_glove.dataset['glove_embedding'])])
# y_true = np.hstack([np.zeros(p_inlier_dl_reuters_glove.dataset.num_rows), np.ones(p_anomaly_dl_reuters_glove.dataset.num_rows)])
# contamination=0.1, kernel='rbf', gamma='scale' ocsvm_args

def One_Class_SVM(embeddings, y_true, verbose=True):

    print("\nOCSVM model training start...")
    start = time.time()
    clf = OCSVM(contamination=0.1, kernel='rbf', gamma='scale')

    clf.fit(embeddings)
    y_pred = clf.predict(embeddings)           
    scores = clf.decision_function(embeddings)
    end = time.time()

    print(f"\nOCSVM model training finish after {end-start} seconds")
    
    if verbose : 
        tsne = TSNE(
            n_components=2,        
            perplexity=30,         
            max_iter=1000,           
            learning_rate=200
        )

        X_tsne = tsne.fit_transform(embeddings)

        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='coolwarm', s=50)
        plt.title("Reuters - OCSVM ")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    return clf, y_pred, scores

    
            
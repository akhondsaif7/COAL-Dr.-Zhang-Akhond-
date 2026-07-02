"""Generate synthetic MQ2008-format .npz files with a learnable linear signal."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings

np.random.seed(42)

N_QUERIES  = 800
N_VAL      = 200
MAX_DOCS   = 40
K          = 8    # actions used by oracle_cb (we generate MAX_DOCS but code crops to K)
N_FEATURES = 47

# Ground-truth linear scoring function: relevance ~ w · feature + noise
w_true = np.random.randn(N_FEATURES)
w_true /= np.linalg.norm(w_true)

def make_split(n_queries, seed_offset=0):
    rng = np.random.RandomState(42 + seed_offset)
    docs_per_query = rng.randint(K, MAX_DOCS + 1, size=n_queries)

    # features: (n_queries, MAX_DOCS, N_FEATURES)
    features = rng.randn(n_queries, MAX_DOCS, N_FEATURES).astype(np.float32)

    # relevances: score = w_true · feature + small noise, then bin into 0-3
    scores = features.reshape(-1, N_FEATURES).dot(w_true)
    scores += 0.3 * rng.randn(n_queries * MAX_DOCS)
    scores = scores.reshape(n_queries, MAX_DOCS)

    # Bin scores into relevance labels 0-3 by quartile
    q25, q75 = np.percentile(scores, 25), np.percentile(scores, 75)
    q50      = np.percentile(scores, 50)
    relevances = np.zeros((n_queries, MAX_DOCS), dtype=np.int32)
    relevances[scores > q25] = 1
    relevances[scores > q50] = 2
    relevances[scores > q75] = 3

    # Zero-pad docs beyond each query's actual count
    for i, d in enumerate(docs_per_query):
        relevances[i, d:] = 0
        features[i, d:]   = 0.0

    return features, relevances, docs_per_query

feat_train, rel_train, docs_train = make_split(N_QUERIES, seed_offset=0)
feat_val,   rel_val,   docs_val   = make_split(N_VAL,     seed_offset=100)

np.savez(settings.BASE_DIR + "data/MQ2008_train.npz",
         relevances=rel_train, features=feat_train, docsPerQuery=docs_train)
np.savez(settings.BASE_DIR + "data/MQ2008_val.npz",
         relevances=rel_val,   features=feat_val,   docsPerQuery=docs_val)

print(f"Generated data/MQ2008_train.npz  ({N_QUERIES} queries, linear signal)")
print(f"Generated data/MQ2008_val.npz    ({N_VAL} queries, linear signal)")

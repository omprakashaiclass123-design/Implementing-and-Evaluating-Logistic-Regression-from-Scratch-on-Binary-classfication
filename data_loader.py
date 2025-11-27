
import numpy as np
def load_data():
    np.random.seed(0)
    X=np.random.randn(500,2)
    y=(X[:,0]+X[:,1]>0).astype(int)
    idx=np.random.permutation(len(X))
    train=idx[:400]; test=idx[400:]
    return X[train],y[train],X[test],y[test]

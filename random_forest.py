from quantum_random_number_genrator import qml_random_choice
from tree import DecisionTree
import numpy as np
from collections import Counter

class random_forest():
    def __init__(self,n_trees=100,max_depth=10,min_sample_split=2,n_features=None ):
        self.n_trees=n_trees
        self.max_depth=max_depth
        self.min_sample_split=min_sample_split
        self.n_features= n_features
        self.trees=[]

    def fit(self,X,y):
        self.trees = []

        for _ in range(self.n_trees):
           tree=DecisionTree(min_samples_split=self.min_sample_split, max_depth=self.max_depth,n_features=self.n_features)
           x_sample,y_sample=self.boot_strap_sample(X,y)
           tree.fit(x_sample,y_sample)
           self.trees.append(tree)

    def boot_strap_sample(self,X,y):
        n_samples=X.shape[0]
        idxs= qml_random_choice(n_samples,n_samples,replace=True)
        return X[idxs],y[idxs]


    def most_common_label(self,y):
        counter=Counter(y)
        most_common=counter.most_common(1)[0][0]
        return most_common

    def predict(self,X):
     predictions=np.array([tree.predict(X)for tree in self.trees])
     tree_pred=np.swapaxes(predictions,0,1)
     predictions=np.array([self.most_common_label(pred) for pred in tree_pred])

     return predictions

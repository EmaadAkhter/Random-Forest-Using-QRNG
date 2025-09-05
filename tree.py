import pennylane as qml
import numpy as np
from quantum_random_number_genrator import qml_random_choice
from collections import Counter

class Node:
    def __init__(self,features = None ,thershold =  None ,left = None ,right = None,*,value = None):
        self.features = features
        self.thershold = thershold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=10,n_features=None):

        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root= None


    def fit(self, X,y):
        self.n_features = X .shape[1] if not self.n_features else min(self.n_features,X.shape[1])
        self.root=self.grow_tree(X,y)


    def grow_tree(self,X,y, depth=0):
        n_samples,n_feats = X.shape
        n_labels=len(np.unique(y))

        if depth>= self.max_depth or n_labels ==1 or  n_samples < self.min_samples_split:
            leaf_val= self.most_common_label(y)
            return Node(value=leaf_val)
        feat_idxs=qml_random_choice(n_feats,self.n_features,replace=False)
        best_features,best_threshold = self.best_split(X,y,feat_idxs)
        left_idxs,right_idxs=self.split(X[:,best_features],best_threshold)

        left=self.grow_tree(X[left_idxs,:],y[left_idxs],depth=depth+1)
        right=self.grow_tree(X[right_idxs,:],y[right_idxs],depth=depth+1)

        return Node(best_features,best_threshold,left,right)


    def best_split(self,X,y,feat_idxs):
        best_gain=-float('inf')
        split_idx, split_threshold= None, None
        for feats_idx in feat_idxs:
             X_column = X[:,feats_idx]
             thresholds=np.unique(X_column)

             for thr in thresholds:
                 gain=self.information_gain(y,  X_column,thr)

                 if gain>best_gain:
                    best_gain=gain
                    split_idx=feats_idx
                    split_threshold=thr

        return split_idx,split_threshold

    def information_gain(self,y, X_column ,thr):
        parent_entropy=self.entrophy(y)

        left_idxs,right_idxs= self.split( X_column,thr)

        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0

        n=len(y)
        n_l,n_r=len(left_idxs),len(right_idxs)
        e_l,e_r=self.entrophy(y[left_idxs]),self.entrophy(y[right_idxs])

        child_entropy=( n_l / n ) * e_l + ( n_r / n ) * e_r

        info_gain=parent_entropy - child_entropy

        return info_gain


    def entrophy(self,y):
       hist = np.bincount(y)
       ps =hist/len(y)
       entropy= -np.sum([p *np.log2(p) for p in ps if p >0])

       return entropy

    def split(self,X_column,thr):
         left_idx=np.argwhere(X_column<=thr).flatten()
         right_idx = np.argwhere(X_column >  thr).flatten()

         return left_idx, right_idx


    def most_common_label(self,y):
        counter=Counter(y)
        most_common=counter.most_common(1)[0][0]
        return most_common

    def predict(self,X):
      return  np.array([self.treverse_tree(x,self.root) for x in X])

    def treverse_tree(self,X,node):
        if node.is_leaf_node():
            return node.value
        if  X[node.features]<=node.thershold:
            return self.treverse_tree(X,node.left)
        else:
            return self.treverse_tree(X,node.right)


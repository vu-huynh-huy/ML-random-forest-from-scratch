import numpy as np
from decisiontree import DecisionTree_Classifier

class RandomForest_Classifier:
    def __init__(self, n_trees = 10, min_samples_split = 2, max_depth = 5, n_features = None, random_state = 42):
        """
        Random Forest Classifier class constructor
        :Input n_trees: int, number of trees
        :Input min_samples_split: int, minimum number of samples to split a node
        :Input max_depth: int, maximum depth of the tree
        :Input n_features: int, number of features
        :Input random_state: int, random state
        :Output self: RandomForest_Classifier
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        self.n_features = n_features
        self.random_state = random_state
        self.trees = []
        np.random.seed(self.random_state)
    def fit(self, X, y):
        """
        Function to train the trees
        :Input X: np.array, features
        :Input y: np.array, labels
        :Output self: RandomForest_Classifier
        """
        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTree_Classifier(min_samples_split = self.min_samples_split, max_depth = self.max_depth, n_features = self.n_features)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        preds = preds.T
        y_pred  = []
        for pred in preds:
            p = np.argmax(np.bincount(pred))
            y_pred.append(p)
        return np.array(y_pred)
    def bootstrap_sample(self, X, y):
        """
        Function to generate bootstrap sample
        :Input X: np.array, features
        :Input y: np.array, labels
        """
        n_sample = np.round(X.shape[0]*2//3)
        idxs = np.random.choice(n_sample, size = n_sample, replace = True)
        return X[idxs], y[idxs]
    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = self.accuracy(y_test, y_pred)
        return accuracy
    def get_params(self):
        return {'n_trees': self.n_trees,
                'min_samples_split': self.min_samples_split,
                'max_depth': self.max_depth,
                'n_features': self.n_features}

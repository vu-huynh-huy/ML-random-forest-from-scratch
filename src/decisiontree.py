import numpy as np

class DecisionTree_Classifier:
    '''
    Implement Decision Tree classifier algorithm
    '''
    def __init__(self, min_samples_split = 2, max_depth = 5, n_features = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features # number of features
    
    def fit(self, X, y):
        '''
        Function to train the tree
        :Input X: np.array, features
        :Input y: np.array, labels
        :Output self: DecisionTree
        '''
        if self.n_features == None:
            self.n_features = X.shape[1]
        else:
            self.n_features = self.n_features
        self.root = self.grow_tree(X, y)
    
    def predict(self, X):
        '''
        Function to predict new data
        :Input X: np.array, features
        :Output y: np.array, predicted classes
    
        '''
        y_predicted = np.array([self.traverse(x) for x in X])
        return y_predicted
    def predict_proba(self, X):
        """
        Function to predict the probability of each class
        :Input X: np.array, features
        :Input node: Node, current node
        :Output y: np.array, predicted classes
        """
        y_preds = []
        y_probs = []
        for x in X:
            y_pred, y_prob = self.traverse(x, proba = True)
            y_preds.append(y_pred)
            y_probs.append(y_prob)
        return np.array(y_preds), np.array(y_probs)

    def traverse(self, x, node = None, proba = False):
        if node is None:
            node = self.root
        if node.is_leaf_node():
            class_pred = node.value
            if proba:
                class_prob = node.best_class_prob
                # percent = node.count_classes/node.lenght
                return class_pred, class_prob
            else:
                return class_pred
        
        if x[node.feature] <= node.threshold:
            return self.traverse(x, node.left, proba = proba)
        else:
            return self.traverse(x, node.right, proba = proba)
    def entropy(self, s):
        '''
        Calculate entropy of a given array
        :Input s: np.array, array of labels
        :Output entropy: float, entropy of the given array
        '''
        #todo
        h = np.bincount(s)/len(s)
        return -np.sum([p*np.log2(p) for p in h if p>0])
    def information_gain(self, y, X_column, threshold):
        '''
        Calculate information gain of a split
        :Input y: np.array, labels
        :Input X_column: np.array, a column of features
        :Input threshold: float, threshold to split the data
        :Output infor_gain: float, information gain
        ''' 
        # TODO
        parent_entropy = self.entropy(y)
        l_indexes, r_indexes = self.split(X_column, threshold)
        if len(l_indexes) == 0 or len(r_indexes) == 0:
            return 0
        else:
            k = len(y)
            l_k = len(l_indexes)
            r_k = len(r_indexes)
            l_entropy = self.entropy(y[l_indexes])
            r_entropy = self.entropy(y[r_indexes])
            child_entropy = (l_k/k)*l_entropy + (r_k/k)*r_entropy
            infor_gain = parent_entropy - child_entropy
            return infor_gain
        
    def split(self, X_column, threshold):
        l_indexes = np.argwhere(X_column <= threshold).flatten()
        r_indexes = np.argwhere(X_column > threshold).flatten()
        return l_indexes, r_indexes
    
    def best_split(self, X, y, feat_indexes):
        '''
        Find the best split given the data
        :Input X: np.array, features
        :Input y: np.array, labels
        :Output best_split: dict, best split
        '''
        # TODO
        best_gain = -1
        split_index, split_threshold = None, None
        # print("feat_idex ", feat_indexes)
        for feat_index in feat_indexes:
            X_column = X[:, feat_index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_index = feat_index
                    split_threshold = threshold
        return {"index": split_index, "threshold": split_threshold}
    
    def grow_tree(self, X, y, depth = 0):
        '''
        Function to grow the tree recursively
        :Input X: np.array, features
        :Input y: np.array, labels
        :Input depth: int, current depth of the tree
        :Output node: Node, current node
        '''
        n_instances, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth > self.max_depth or n_instances < self.min_samples_split or n_labels == 1):
            leaf_value = self.most_common_label(y)
            return Node(value = leaf_value)

        # choose random subset of features
        feat_indexes = np.random.choice(n_features, self.n_features, replace = False)

        # select the best split
        best_split = self.best_split(X, y, feat_indexes)
        l_indexes, r_indexes = self.split(X[:, best_split["index"]], best_split["threshold"])
        left = self.grow_tree(X[l_indexes,:], y = y[l_indexes], depth = depth + 1)
        right = self.grow_tree(X[r_indexes,:], y = y[r_indexes], depth = depth + 1)
        node = Node(best_split["index"], best_split['threshold'], left, right, lenght = len(y), label = y)
        return node 
    def most_common_label(self, y):
        '''
        Find the most common label
        :Input y: np.array, labels
        :Output label: int, the most common label
        '''
        label = np.bincount(y).argmax()
        return label    


class Node:
    '''
    Node class for Decision Tree
    '''
    def __init__(self, feature = None, threshold = None, left = None,
                 right = None, gain = None, value = None, lenght = None, label = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value
        self.n_classes, self.count_classes  = np.unique(label, return_counts = True)
        self.best_class = self.n_classes[np.argmax(self.count_classes)]
        self.best_class_prob = self.count_classes[np.argmax(self.count_classes)]/np.sum(self.count_classes)
        self.lenght = lenght

    def is_leaf_node(self):
        '''
        Check if the node is leaf node
        :Output bool: True if the node is leaf node, False otherwise
        '''
        if self.value is not None:
            return True
        else:
            return False
        
    def print_tree(self, spacing = ""):
        '''
        Print the tree
        :Input spacing: str, spacing between nodes
        '''
        if self.is_leaf_node():
            print(spacing + "Predict", self.value)
            return
        print(spacing + "Is", self.feature, "<=", self.threshold, "?")
        print(spacing + "--> True:")
        self.left.print_tree(spacing + "  ")
        print(spacing + "--> False:")
        self.right.print_tree(spacing + "  ")
    

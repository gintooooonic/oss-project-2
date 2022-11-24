# PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/gintooooonic/oss-project-2

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_dataset(dataset_path):
    return pd.read_csv(dataset_path)

def dataset_stat(dataset_df):	
	n_feats = dataset_df.shape[1]
	target_size = dataset_df.groupby('target').size()
	return (n_feats, target_size[0], target_size[1])

def split_dataset(dataset_df, testset_size):
	X = dataset_df.drop(columns='target', axis=1)
	y = dataset_df['target']
	return train_test_split(X, y, test_size=testset_size)

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	cls = DecisionTreeClassifier()
	cls.fit(x_train, y_train)

	acc = accuracy_score(cls.predict(x_test), y_test)
	prec = precision_score(cls.predict(x_test), y_test)
	recall = recall_score(cls.predict(x_test), y_test)

	return acc, prec, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	cls = RandomForestClassifier()
	cls.fit(x_train, y_train)

	acc = accuracy_score(cls.predict(x_test), y_test)
	prec = precision_score(cls.predict(x_test), y_test)
	recall = recall_score(cls.predict(x_test), y_test)

	return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test):
	cls = SVC()
	cls.fit(x_train, y_train)

	acc = accuracy_score(cls.predict(x_test), y_test)
	prec = precision_score(cls.predict(x_test), y_test)
	recall = recall_score(cls.predict(x_test), y_test)

	return acc, prec, recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)

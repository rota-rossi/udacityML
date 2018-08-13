#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
# features_list = ['poi', 'salary', 'bonus', 'exercised_stock_options',
#   'shared_receipt_with_poi', 'from_poi_percentage', 'to_poi_percentage']
features_list = ['poi', 'salary', 'bonus', 'exercised_stock_options']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# dataset_len = 0
# total_poi = 0
# total_non_poi = 0
# total_nan = 0
# max_attributes_len = 0
# for (user, attributes) in data_dict.iteritems():
#   dataset_len += 1
#   max_attributes_len = len(attributes.items()) if len(attributes.items()) > max_attributes_len else max_attributes_len
#   if attributes['poi'] == 1:
#     total_poi += 1
#   else: 
#     total_non_poi += 1
#   for (k, v) in attributes.iteritems():
#     if v == 'NaN':
#       total_nan += 1

# print ('Dataset Size: ', dataset_len)
# print ('Total Poi: ', total_poi)
# print ('Total Non Poi: ', total_non_poi)
# print ('Total NaN: ', total_nan)
# print ('Max Attributes: ', max_attributes_len)


# Task 2: Remove outliers

# The first obvious outlier to be removed is the 'TOTAL' record 
total = data_dict.pop('TOTAL', 0)

# after that, we should remove every record that does not have any information about salary
data_dict = dict((k, v) for (k, v) in data_dict.iteritems() if v['salary'] != 'NaN')



# Task 3: Create new feature(s)

new_dict = {}


for (person, details) in data_dict.iteritems():
  # Commented manipulation for possible future analysis.

  # details['salary'] = 1.0 * details['salary'] / total['salary']
  # details['bonus'] = 1.0 * details['bonus'] / total['bonus'] if details['bonus'] != 'NaN' else 0.0
  # details['exercised_stock_options'] = 1.0 * details['exercised_stock_options'] / total['exercised_stock_options'] if details['exercised_stock_options'] != 'NaN' else 0.0

  if details['from_this_person_to_poi'] != 'NaN' and details['from_messages'] != 'NaN':
    details['from_poi_percentage'] = 1.0 * details['from_this_person_to_poi'] / details['from_messages']
  else:
    details['from_poi_percentage'] = .0
  if details['from_poi_to_this_person'] != 'NaN' and details['to_messages'] != 'NaN':
    details['to_poi_percentage'] = 1.0 * details['from_poi_to_this_person'] / details['to_messages']
  else:
    details['to_poi_percentage'] = .0
  new_dict[person] = details


# Store to my_dataset for easy export below.
my_dataset = new_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)



# MinMaxScaler - It was not necessary, as PCA did some sort of feature scaling

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# features = scaler.fit_transform(features)



# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script.


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from time import time


from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


pca = PCA(n_components=3)
fitted_pca = pca.fit(features_train)
features_train = fitted_pca.transform(features_train)
features_test = fitted_pca.transform(features_test)


# Code to show the features list and their respective variances
#
# for ii in range(len(pca.components_)):
#   for jj in range(len(pca.components_[ii])):
#     print features_list[jj+1], ' - ', pca.components_[ii][jj]


# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script.

# Parameter Selection

# The commented code was used to properly select the parameters for 
# the KNeighborsClassifier algorithm

# from sklearn.model_selection import GridSearchCV
# parameters = {
#   'algorithm': ['auto'],
#   'n_neighbors': [1, 3, 5, 10],
#   'leaf_size': [5, 10, 20, 30, 50],
#   'weights': ['uniform', 'distance'],
#   'p': [1, 2]
# }
# estimator = GridSearchCV(KNeighborsClassifier(), parameters)
# estimator = estimator.fit(features_train, labels_train)
# print "Best estimator found by grid search:"
# print estimator.best_estimator_

# estimator = estimator.best_estimator_


# estimator = GaussianNB()


estimator = KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='minkowski',
                                 metric_params=None, n_jobs=1, n_neighbors=3, p=1,
                                 weights='uniform')

clf = Pipeline([
    ('pca', pca),
    ('clf', estimator),
])

# When testing without PCA, comment the pipeline above and uncomment the line below.
#
# clf = estimator

clf = clf.fit(features_train, labels_train)


# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

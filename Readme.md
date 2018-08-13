#### ANSWERS FOR THE QUESTIONS ####

1. Summarize for us the goal of this project and how machine learning is useful in trying to 
accomplish it. As part of your answer, give some background on the dataset and how it can be used to
answer the project question. Were there any outliers in the data when you got it, and how did you 
handle those?

The project goal is to identify Persons of Interest (POI) in the Enron email messages dataset. As 
the dataset is huge (>500,000 messages), it is quite complicated for a human to analyse all those 
data, so a machine learning classifier may be helpful in this sense. 

The dataset comprises several features from the employees from Enron. Among those features, 
we have salaries, bonuses, number of emails exchanged with probable POI, and also if the person is
a POI themself.

There are a few outliers in the dataset - the main one is a dummy employee record named 'TOTAL',
which is, as the name says, the accumulator for the others records. Also, we have some employee 
records that do not contain a salary information (possibly external consultants). I opted by 
removing those outliers.

Some statistics in the dataset (before treatment):
- Dataset Size: 146;
- Total Person of Interest: 18;
- Total Non POI: 128;
- Total Invalid datapoints (NaN): 1358;
- Total Features: 21;

After data treatment (removal of outliers):

- dataset length: 94;
- Total Person of Interest: 17;
- Total Non POI: 77;
- Existing features used: 9 (5 directly and 4 indirectly);
- Generated features: 2;

2. What features did you end up using in your POI identifier, and what selection process did you use 
to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should 
attempt to engineer your own feature that does not come ready-made in the dataset -- explain what 
feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in 
the final analysis, only engineer and test it.) In your feature selection step, if you used an 
algorithm like a decision tree, please also give the feature importances of the features that you 
use, and if you used an automated feature selection function like SelectKBest, please report the 
feature scores and reasons for your choice of parameter values. 

After testing several features, I decided by using the following ones (and the reasons why): 
- poi -> if the person is a Person of Interest;
- salary -> it is expected that the POI involved in the fraud were in a high position, which is
related to high salaries;
- bonus -> One of the documented ways that the POI's used to make money from the fraud was to 
artifitially inflate the company profits, and take huge bonuses from it;
- exercised_stock_options -> right before the company went bankrupt, several high profile employees
exercised their stock options;
- shared_receipt_with_poi: emails that the person received that were also sent to a POI.

I also created the features 'to_messages', 'from_poi_to_this_person', 'from_messages', and 
'from_this_person_to_poi' to generate the following features:

- from_poi_percentage: generated from `from_this_person_to_poi / from_messages`;
- to_poi_percentage: generated from `from_poi_to_this_person / to_messages`;

Both measures are a good indicator of the level of involvement each person had with persons of 
interest, potentially showing that they are a POI themselves. Initially I tought that those new
features would be a good fit for the prediction, but I was mistaken (as can be seen below):

I used Principal component analysis (PCA) for dimension reduction - reducing it to 3 components 
greatly improved the algorithm performance. 

As we can see in the variances shown below, the PCA reduction also did some kind of dimension 
scaling. This way, I didn't have to use any scalers for this purpose (using `MinMaxScaler`
game me the same final performance): 

- component 1:
  
  salary:  0.0203304605566  
  bonus:  0.134478159961  
  exercised_stock_options:  0.990707977234  
  shared_receipt_with_poi:  2.66918293018e-05  
  from_poi_percentage:  2.51425329156e-09  
  to_poi_percentage:  3.67443837924e-10  

- component 2: 
  salary:  0.0379163022718  
  bonus:  0.990096012215  
  exercised_stock_options:  -0.13517319607  
  shared_receipt_with_poi:  0.000669090863749  
  from_poi_percentage:  1.48071538766e-08  
  to_poi_percentage:  1.37641880955e-09  

- component 3:

  salary:  0.9990740457  
  bonus:  -0.040312292651  
  exercised_stock_options:  -0.0150301800654  
  shared_receipt_with_poi:  0.000252897419622  
  from_poi_percentage:  -1.2900067922e-07  
  to_poi_percentage:  -6.8375386973e-08  


We can see that the features `shared_receipt_with_poi`, `from_poi_percentage` and `to_poi_percentage`
have very little effect in the generation of all components. Because of that, I opted by removing them
from the feature list. 

Using only a `MinMaxScaler`, without a PCA, I reached the following performance:

Accuracy: 0.83440       
Precision: 0.66996      
Recall: 0.33900  
F1: 0.45020

I opted by keeping using the PCA instead of the scaler for a more balanced performance between 
precision and recall.

3. What algorithm did you end up using? What other one(s) did you try? How did model performance 
differ between algorithms?  

I tested several algorithms, trying to balance `precision` and `recall`. Among them:
- GaussianNB;
- LinearSVC;
- SVC(kernel='rbf');
- RandomForestClassifier;
- SGDClassifier;
- AdaBoostClassifier;
- DecisionTreeClassifier;

Using GaussianNB, the statistics were:

  Accuracy: 0.79090       
  Precision: 0.44728      
  Recall: 0.19300  
  F1: 0.26965    

At the end, I used KNeighborsClassifier, which gave me a balanced precision and recall levels. 
The Statistics for the KNeighborsClassifier were:

Accuracy: 0.82060  
Precision: 0.57875  
Recall: 0.37850   
F1: 0.45768   

As we can see, the overall performance for KNeighborsClassifier algorithm is better than GaussianNB.

4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do 
this well?  How did you tune the parameters of your particular algorithm? What parameters did you 
tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the 
one you picked, identify and briefly explain how you would have done it for the model that was not 
your final choice or a different model that does utilize parameter tuning, e.g. a decision tree 
classifier). 

An algorithm can use several options to be customized to your dataset. These parameters allow the 
user to improve the algorithm capacity to properly classify the data. 

To tune the selected algorithm, I used the `GridSearchCV` selector. This algorithm does an 
exhaustive grid search in the supplied hyper-parameters and returns the best configuration found. 
I used the following hyper-parameters for `KNeighborsClassifier`:

```python
 parameters = {
   'algorithm': ['auto'],
   'n_neighbors': [1, 3, 5, 10],
   'leaf_size': [5, 10, 20, 30, 50],
   'weights': ['uniform', 'distance'],
   'p': [1, 2]
 }
```

These parameters were based on the Classifier documentation, on some examples observation and 
(a lot of) guesswork. Based on that, the `GridSearchCV` selected the following parameters:

```python
algorithm='auto', 
leaf_size=5, 
metric='minkowski',
metric_params=None, 
n_jobs=1, 
n_neighbors=3, 
p=1,
weights='uniform'
```

5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you 
validate your analysis? 

To guarantee that your algorithm is properly trained and not overfit, it is important to separate
some data for testing after your algorithm is trained. This data is then used to validate if your 
code is properly configured and trained to predict results in samples that it did not have seen 
before.

A common mistake is to forget to separate a test set, and train on your entire dataset. Then, when
it is time to validate, the algorithm performance is usually bad (because of overfitting).

I separate a percentage of the data (30%) as a test dataset, and after training the algorithm, I 
used it to validate if the predictions were properly being done.

6. Give at least 2 evaluation metrics and your average performance for each of them.  
Explain an interpretation of your metrics that says something human-understandable about your 
algorithm’s performance. 

Accuracy: 0.82060
This is a metric that shows the percentage of the test data that is properly being categorized.

Precision: 0.57875
Out of all the items labeled as POI, how many are truly POI.

Recall: 0.37850
Out of all the persons that are truly POI, how many were correctly classified as positive. 


#### REFERENCES ####

1. Scikit Learn: http://scikit-learn.org/stable/index.html
2. matplotlib: https://matplotlib.org/index.html
3. Wikipedia: https://en.wikipedia.org/wiki/F1_score

I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.
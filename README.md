# -DECISION-TREE-IMPLEMENTATION

 *COMPANY*: CODTECH IT SOLUTIONS

 *NAME*: Darshan RH
 
 *INTERN ID*: CT04DG2374
 
 *DOMAIN*: MACHINE LEARNING
 
 *DURATION*: 4 WEEKS
 
 


 #  Task 1 – Decision Tree Classifier (CodTech ML Internship)

This repository contains the complete solution for **Task 1** of the CodTech Machine Learning Internship. The objective of this task is to build and visualize a **Decision Tree Classifier** using the popular Scikit-Learn library on a standard dataset.we used the jupyter lab using anaconda to complete this task

---

## Objective

The goal of this task is to:
- Understand the fundamentals of decision trees.
- Train a classifier using a real-world dataset.
- Visualize the structure of the decision tree.
- Evaluate the performance of the model using metrics such as accuracy and confusion matrix.

This is one of the most basic yet powerful supervised machine learning algorithms used for classification and regression problems.

---

##  Dataset Used

We used the **Iris dataset**, which is built into Scikit-Learn. This dataset is widely used for pattern recognition and classification tasks.

- Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Target Labels:
  - Setosa
  - Versicolor
  - Virginica

It contains 150 samples, evenly distributed across the three classes.
### Parameters:

riterion: It measure the quality of a split. Supported values are 'gini', -'entropy' and 'log_loss'. The default value is 'gini'
 
splitter: This parameter is used to choose the split at each node. Supported values are 'best' & 'random'. The default value is 'best'
 
max_features: It defines the number of features to consider when looking for the best split.

max_depth: This parameter denotes maximum depth of the tree (default=None).

min_samples_split: It defines the minimum number of samples reqd. to split an internal node (default=2).

min_samples_leaf: The minimum number of samples required to be at a leaf node (default=1)

max_leaf_nodes: It defines the maximum number of possible leaf nodes.

min_impurity_split: It defines the threshold for early stopping tree growth.

class_weight: It defines the weights associated with classes.

ccp_alpha: It is a complexity parameter used for minimal cost-complexity pruning

---
### Requirements
Make sure you install the following libraries before running the notebook:

```python
pip install numpy pandas matplotlib scikit-learn
```

## Steps Performed

### 1. Importing libraries
```python
import pandas as pd  
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
%matplotlib inline
```
We will import libraries like Scikit-Learn ,pandas  and matploitlib for machine learning decision tree

### 2. Data Loading
```python
from sklearn.datasets import load_iris
iris = load_iris()
```
In order to perform classification load a dataset. For demonstration one can use sample datasets from Scikit-Learn such as Iris

### 3:Splilt Dataset
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
```
we used train test split method from sklearn.model_selection to split the dataset into training set and test data split that 75% and 25% respectively as we used test_size=0.25

### 4: Model defining and plotting 
```python
model = DecisionTreeClassifier(max_depth=2)# set the max depth =2 as we dont need it anymore as we analyse in model tree
model.fit(X_train, y_train)
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=iris.feature_names, filled=True)
plt.show()
```

decision tree model ![Image](https://github.com/user-attachments/assets/a6653874-67e3-40c1-9370-e365e26b33b5)
this decision tree model has 2 nodes as we can see and accuracy of 0.98 approx

This section visualizes the decision tree. It clearly shows how the data is split at each node, including:

-Feature names used

-Threshold values

-Gini index

-Class distribution

-Predicted class

### 5: Making Predictions & Evaluating the Model

```python
y_pred=model.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report
core=accuracy_score(y_pred, y_test)
print(score)
print(classification_report(y_pred,y_test))
```

### Summary and Reflections

This task helped me get a hands-on understanding of:
-How decision trees work

-How to use Scikit-Learn to quickly build and train a model

-How to split data properly and evaluate results

-How to visualize the internal logic of a tree-based model

While this was a basic task, I now understand how models "decide" by creating split conditions. It also made me more comfortable working with Jupyter Notebooks and Scikit-Learn.

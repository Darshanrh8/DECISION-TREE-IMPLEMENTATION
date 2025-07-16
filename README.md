# 🔍 CodTech ML Internship – Task 1: Decision Tree Classifier

**COMPANY:** CodTech IT Solutions  
**NAME:** Darshan RH  
**INTERN ID:** CT04DG2374  
**DOMAIN:** Machine Learning  
**DURATION:** 4 Weeks

---

## 🎯 Objective

- Understand the fundamentals of decision trees  
- Train a classifier using a real-world dataset  
- Visualize the structure of the decision tree  
- Evaluate the performance of the model using accuracy and classification metrics  

---

## 📚 Dataset Used: Iris Dataset

The Iris dataset is built into Scikit-Learn.  
It contains 150 samples evenly distributed across 3 classes:  
- Setosa  
- Versicolor  
- Virginica  

**Features:**
- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

---

## ✅ Requirements

Run this cell to install required libraries:

!pip install -q numpy pandas scikit-learn matplotlib

pgsql
Copy
Edit

---

## 🔧 Step 1: Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

%matplotlib inline

📥 Step 2: Load the Dataset
python
Copy
Edit
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for better readability
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df.head()

🔀 Step 3: Split the Dataset
python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


🌳 Step 4: Define the Model & Visualize the Tree
python
Copy
Edit
# Create and train the model
model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(10, 6))
plot_tree(model, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names,
          filled=True)
plt.title("Decision Tree Visualization")
plt.show()

📊 Step 5: Make Predictions & Evaluate the Model
python
Copy
Edit
# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
score = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f"Accuracy: {score:.2f}")
print("\nClassification Report:\n", report)





🧠 Summary and Reflections
This task helped me:

Understand how decision trees work

Learn to use Scikit-Learn for training and evaluating models

Practice splitting datasets properly

Visualize the internal structure of a decision tree

Gain confidence working with Jupyter/Colab environments

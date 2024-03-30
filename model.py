from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create and train Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Save the model to a file
joblib.dump(clf, 'iris_decision_tree_model.joblib')

import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Iris Dataset
iris = load_iris()
X = iris.data 
y = iris.target  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=200) 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("-" * 30)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


filename = 'iris_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Success! Model saved as '{filename}'")

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = CascadeForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred) * 100
print("Accuracy: {:.3f} %".format(acc))

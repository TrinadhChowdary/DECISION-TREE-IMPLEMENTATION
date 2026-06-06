import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix

iris=load_iris()
X=iris.data
Y=iris.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

model=DecisionTreeClassifier(criterion="gini",max_depth=4,random_state=42)

model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

accuracy=accuracy_score(Y_test,Y_pred)

print("Actual Outcomes:")
print(Y_test)
print()
print("\nPredicted Outcomes:")
print(Y_pred)
print()
print("Accuracy:",accuracy)
print()
print()
print("Confusion Matirx :")

print(confusion_matrix(Y_test,Y_pred))
plt.figure(figsize=(14,8))
plot_tree(model,feature_names=iris.feature_names,class_names=iris.target_names)

plt.title("Decision-Tree Classifier -Iris Dataset")

plt.show()

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def preprocessData(data):

    print(data)
    # In this setp we fill missing 'Age' values with mean value of the column by using SimpleImputer
    # SimpleImputer is a preprocessing method used to fill in missing values in a dataset
    imputer1 = SimpleImputer(strategy='mean')
    data['Age'] = imputer1.fit_transform(data['Age'].values.reshape(-1, 1))

    # In this step we are converting categorical data into numerical data(Label Encoding) for the columns Sex and Embarked
    # This is to ensure that we that use the transformed data in this column as input for our model
    encoder = LabelEncoder()
    data['Sex'] = encoder.fit_transform(data['Sex'])
    data['Embarked'] = encoder.fit_transform(data['Embarked'])

    # In this setp we fill missing 'Embarked' values with most frequent value of the column by using SimpleImputer
    imputer2 = SimpleImputer(strategy='most_frequent')
    data['Embarked'] = imputer2.fit_transform(data['Embarked'].values.reshape(-1, 1))

    data = data.drop_duplicates()
    print(data)
    return data


# Start HERE
training_data = preprocessData(pd.read_csv("train.csv"))
test_data = preprocessData(pd.read_csv("test.csv"))

# Step 2: Feature Selection
# From the data we can infer that Name,Cabin,Ticket does not effect the information and 
# features are not useful. Hence we can remove those columns

# We calculate the absolute correlation between each feature and the 'Survived' column. The corr() function computes the correlation matrix, and abs() is applied to get the absolute values of the correlation coefficients.

# We specify 'n' to determine the number of features to select based on the highest absolute correlation values.

# We select the top-n features with the highest absolute correlation using nlargest(n).

# Finally, we obtain the names of the selected features as 'selected_feature_names'.
correlation = training_data.drop(["Name","Cabin","Ticket","Fare"],axis=1).corr().abs()['Survived']

n=6
selected_features = correlation.nlargest(n).index

# Get the names of the selected features
selected_feature_names = list(selected_features)
selected_feature_names.remove('Survived')
# Loading data for those selected features
X_train = training_data[selected_feature_names]
Y_train = training_data['Survived']
X_test = test_data[selected_feature_names]


# Step 3: Learn and Fine-Tune a Decision Tree Model
decision_tree = DecisionTreeClassifier(min_impurity_decrease = 0.001)
decision_tree.fit(X_train, Y_train)
# Export the decision tree to a Graphviz .dot file
tree_data = export_graphviz(decision_tree, out_file='tree_data', filled=True, feature_names=X_train.columns, class_names=['Not Survived', 'Survived'])

# Visualize the file with Graphviz
tree_plot = graphviz.Source.from_file('tree_data')
tree_plot.render("decision_tree")  # Save as PDF or other formats

predicted_output = decision_tree.predict(X_test)
predicted_output_data = pd.DataFrame({"PassengerId":test_data.PassengerId,"Survived":predicted_output})
print(predicted_output_data)
predicted_output_data.to_csv("prediction.csv", index=False)

# Step 4: Cross-Validation for Decision Tree
decision_tree_accuracy = cross_val_score(decision_tree, X_train, Y_train, cv=5, scoring='accuracy')
average_accuracy = decision_tree_accuracy.mean()
print("Average Decision Tree Accuracy:", average_accuracy)

# Step 5: Cross-Validation for Random Forest
random_forest = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16)
random_forest.fit(X_train, Y_train)

random_forest_accuracy = cross_val_score(random_forest, X_train, Y_train, cv=5, scoring='accuracy')
average_accuracy_rf = random_forest_accuracy.mean()
print("Average Random Forest Accuracy:", average_accuracy_rf)


# Bagging Classifier

bagging_classifier = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=200, random_state=42)
bagging_classifier.fit(X_train,Y_train)
# Apply cross-validation to estimate the average classification accuracy
bagging_classifier_accuracy = cross_val_score(bagging_classifier, X_train, Y_train, cv=5, scoring='accuracy')
average_accuracy_bagging = bagging_classifier_accuracy.mean()
print("Average Bagging Accuracy:", average_accuracy_bagging)

# Adaboost classifier

adaboost_classifier = AdaBoostClassifier(DecisionTreeClassifier(random_state=42), n_estimators=200, random_state=42, learning_rate=0.6)
adaboost_classifier.fit(X_train,Y_train)
# Apply cross-validation to estimate the average classification accuracy
adaboost_classifier_accuracy = cross_val_score(adaboost_classifier, X_train, Y_train, cv=5, scoring='accuracy')
average_accuracy_adaboost = adaboost_classifier_accuracy.mean()
print("Average AdaBoost Accuracy:", average_accuracy_adaboost)

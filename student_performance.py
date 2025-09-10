import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder


# Loading dataset
students = pd.read_csv("/Users/harshsrivastava/Downloads/Students Performance .csv")
print(students.head())
students = students.drop(columns=['Transportation', 'Student_ID', 'Student_Age','Scholarship'])

# Drop rows where 'Listening_in_Class' or 'Notes' has invalid value 6
students = students[students['Listening_in_Class'].isin(['Yes','No'])]
students = students[students['Notes'].isin(['Yes','No'])]
students = students[students['Attendance'].isin(['Sometimes','Always','Never'])]

X = students.drop(columns=['Grade']).copy()  # Features
y = students['Grade'].copy()                 

# Boys vs Girls vs All
boys = students[students['Sex'] == 'Male'].copy()
girls = students[students['Sex'] == 'Female'].copy()

######################################
# Analysis
######################################

# Insights
print(students.groupby('Grade')['Sex'].count())

# Failed students
plt.title("Failed students")
fail_boys = boys[boys['Grade'] == 'Fail'].shape[0]
fail_girls = girls[girls['Grade'] == 'Fail'].shape[0]
plt.bar(['Boys', 'Girls'], [fail_boys, fail_girls], color=['blue', 'pink'])
plt.show()

# Gender
plt.title("Gender")
plt.pie([len(boys['Project_work']),len(girls['Project_work'])], labels = ['Boys', 'Girls'], autopct='%.4f%%')
plt.show()


# High_School_Type
plt.title("High_School_Type : BOYS")
distinct_values_boys = boys['High_School_Type'].unique().tolist()
plt.bar(distinct_values_boys, boys.groupby('High_School_Type')['Sports_activity'].count())
plt.show()
plt.title("High_School_Type : GIRLS")
distinct_values_girls = girls['High_School_Type'].unique().tolist()
plt.bar(distinct_values_girls, girls.groupby('High_School_Type')['Sports_activity'].count(), color= 'pink')
plt.show()

preprocessing = ColumnTransformer(
    transformers= [
        ("Sex", OrdinalEncoder(categories=[['Male', 'Female']]), ["Sex"]), 
        ("High_School_Type", OrdinalEncoder(categories=[['State', 'Other','Private']]), ["High_School_Type"]), 
        ("Additional_Work", OrdinalEncoder(categories=[['Yes', 'No']]), ["Additional_Work"]), 
        ("Sports_activity", OrdinalEncoder(categories=[['Yes', 'No']]), ["Sports_activity"]), 
        ("Attendance", OrdinalEncoder(categories=[['Sometimes', 'Always','Never']]), ["Attendance"]), 
        ("Reading", OrdinalEncoder(categories=[['Yes', 'No']]), ["Reading"]), 
        ("Notes", OrdinalEncoder(categories=[['Yes', 'No']]), ["Notes"]), 
        ("Listening_in_Class", OrdinalEncoder(categories=[['Yes', 'No']]), ["Listening_in_Class"]), 
        ("Project_work", OrdinalEncoder(categories=[['Yes', 'No']]), ["Project_work"]), 
    ], remainder='passthrough'
)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                 # Dataset (X = features, y = labels)
    test_size=0.1,        # 20% of data is for testing, 80% for training
    random_state=42       # Seed value for reproducibility (same split every run)
)
X_train_transformed = preprocessing.fit_transform(X_train)
X_test_transformed = preprocessing.transform(X_test)



# Finding the best 5 features
select_k_best = SelectKBest(score_func=chi2, k=5)
X_train_k_best = select_k_best.fit_transform(X_train_transformed, y_train)

# Get names of selected features
feature_names = X_train.columns  # original feature names
selected_features = feature_names[select_k_best.get_support()]
print("Selected features:", selected_features)

# Encoding the output
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.values.reshape(-1,1))
y_test = encoder.fit_transform(y_test.values.reshape(-1,1))

# Doing PCA
feature_names = preprocessing.get_feature_names_out()
print("Transformed feature names:", feature_names)
selected_features = ['Project_work','Listening_in_Class','Notes','Reading','Attendance','Weekly_Study_Hours']
selected_indices = [i for i, f in enumerate(feature_names) if any(sf in f for sf in selected_features)]
pca = PCA(n_components=3, random_state=42)
X_pca_boost = pca.fit_transform(X_train_transformed[:, selected_indices])
X_pca_boost_test = pca.transform(X_test_transformed[:, selected_indices])

# Training and predicting values
from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=100,    # Number of boosting rounds
    learning_rate=0.1,   # Step size shrinkage
    max_depth=None,         # Max tree depth
    subsample=1.0,       # Fraction of samples per tree
    colsample_bytree=1.0,# Fraction of features per tree
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
xgb.fit(X_pca_boost, y_train)
y_pred = xgb.predict(X_pca_boost_test)

# Model evaluation
print("xgb Regression Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))




# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
# dataset_path = "banksim_data.csv"
dataset_url = "https://drive.google.com/file/d/1ya8I-xn0sdjDN4MJR-rvqODk-lEtFGEh/view?usp=sharing"
file_id = dataset_url.split("/")[5]
download_url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(download_url)
# print(df.head())

print("Unique zipCodeOri values: ", df.zipcodeOri[1])
print("Unique zipMerchant values: ", df.zipMerchant[1])

# Dropping zipCodeOri and zipMerchant as it only has 1 unique value
df = df.drop(['zipcodeOri','zipMerchant'],axis=1)

# Print the updated DataFrame to verify the changes
print(df.head())

# Fraud and Non-fraud Dataset
fraud_data = df.loc[df.fraud == 1]
nonfraud_data = df.loc[df.fraud == 0]

# Visualize comparison of normal and fraudulent payments 
# sns.countplot(x="fraud", data=df)
# plt.title("Count of Fraudulent Payments")
# plt.show()
# print("Number of normal examples: ",nonfraud_data.fraud.count())
# print("Number of fradulent examples: ",fraud_data.fraud.count())

# Explore mean and fraud percentage grouped by category
# print("Mean category grouped by category: \n", df.groupby("category")["amount", "fraud"].mean())
# grouped_fraud = fraud_data.groupby("category")["amount"]
# grouped_nonfraud = nonfraud_data.groupby("category")["amount"]
# category_compare_data = pd.concat([grouped_fraud.mean(), grouped_nonfraud.mean(), df.groupby("category")["fraud"].mean()*100], keys=["Fraudulent",
# "Non-Fraudulent", "Percentage (in %)"], axis=1, sort=False).sort_values(by=["Fraudulent"])
# print(category_compare_data)

# Visualize fraud and non-fraud data using boxplot
# plt.figure(figsize=(40,10))
# sns.boxplot(x=df.category,y=df.amount)
# plt.title("Boxplot for the Amount spend in category")
# plt.ylim(0,4000)
# plt.xticks(rotation=45, ha='right', fontsize=8)
# plt.legend()
# plt.show()
# print("Insights: Average amount spent per category is similar, except for the amount of fraud for the travel category which spikes drastically.")

# Visualize amounts of fraud and non-fraud using histogram
plt.hist(fraud_data.amount, alpha=0.5, label='Fraud',bins=100)
plt.hist(nonfraud_data.amount, alpha=0.5, label='Non-fraud',bins=100)
plt.title("Histogram for Fraudulent and Non-fraudulent payments")
plt.ylim(0,10000)
plt.xlim(0,1000)
plt.legend()
plt.show()
print("Insights: Fraudulent payments are less in frequency/count but larger in amount.")

# Visualize data based on Age Groups
mean_age_data = (df.groupby('age')['fraud'].mean()*100).reset_index()
compare_age_data = mean_age_data.rename(columns={'age' : 'Age', 'fraud' : 'Fraud Percentage'}).sort_values(by="Fraud Percentage")
print(compare_age_data)
print("Insights: Fraud happens most often on the 0-th category which is the under 18 age group.")

# Changing categorical features into dummies
df.loc[:,['customer','merchant','category']].astype('category') 
df_dummy = pd.get_dummies(df.loc[:,['customer','merchant','category','gender']],drop_first=True) 
# print(df_dummy.info())

# Changing categorical 
col_categorical = df.select_dtypes(include= ['object']).columns #categorical = object
for col in col_categorical:
    df[col] = df[col].astype('category')
df[col_categorical] = df[col_categorical].apply(lambda x: x.cat.codes)
print(df.head())

# Determine X and Y variable
X = df.drop(['fraud'],axis=1)
y = df['fraud']
print(X.head(),"\n")
print(y.head())

# Balancing sample set using SMOTE 
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
y_res = pd.DataFrame(y_res, columns=['fraud'])  # Assigning column name 'fraud' to y_res
# print(y_res['fraud'].value_counts())

# Execute train test split
y_res = y_res.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42, shuffle=True, stratify=y_res)

# Perform cross-validation
lr_clf = LogisticRegression() #use logistics regression classifier
cv_scores = cross_val_score(lr_clf, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Plot a ROC_AUC curve to determine performance of classification
def roc_auc_curve(y_test, preds):
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

preds = lr_clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
roc_auc_curve(y_test, preds)

# Calculate base score
base_score = nonfraud_data.fraud.count() / np.add(nonfraud_data.fraud.count(), fraud_data.fraud.count()) * 100
print("Base score is:", base_score)

# K-Neighbours Classifier
knn = KNeighborsClassifier(n_neighbors=5,p=1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("Classification Report for K-Nearest Neighbours: \n", classification_report(y_test, y_pred))
print("Confusion Matrix of K-Nearest Neigbours: \n", confusion_matrix(y_test,y_pred))
roc_auc_curve(y_test, knn.predict_proba(X_test)[:,1])

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42,verbose=1,class_weight="balanced")

rf_clf.fit(X_train,y_train)
y_pred = rf_clf.predict(X_test)

print("Classification Report for Random Forest Classifier: \n", classification_report(y_test, y_pred))
print("Confusion Matrix of Random Forest Classifier: \n", confusion_matrix(y_test,y_pred))
roc_auc_curve(y_test, rf_clf.predict_proba(X_test)[:,1])




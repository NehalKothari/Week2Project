import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,f1_score,roc_auc_score,roc_curve,recall_score,confusion_matrix,classification_report
df=pd.read_csv("creditcard.csv")
print("\n Dataset Info:")
print(df.info())
print("\n Missing Values:")
print(df.isnull().sum())
print("\n Shape Before Handling:")
print(df.shape)
num_col=df.select_dtypes(include="number").columns
cat_col=df.select_dtypes(include="object").columns
for col in cat_col:
    df[col]=df[col].fillna(df[col].mode()[0])
for col in num_col:
    df[col]=df[col].fillna(df[col].median())
print("\n Missing Values:")
print(df.isnull().sum())
print("\n Dupllicate Values:",df.duplicated().sum())
df=df.drop_duplicates()
q1=df["Amount"].quantile(0.25)
q3=df["Amount"].quantile(0.75)
IQR=q3-q1
lower=q1-1.5*IQR
upper=q3+1.5*IQR
outliers=df[(df["Amount"]<lower)|(df["Amount"]>upper)]
print("\n No of Outliers:",len(outliers))
print(df["Class"].value_counts())
sns.countplot(x="Class",data=df)
plt.savefig("images/data_balanced.png")
plt.close()
plt.show()
feature_names=df.drop("Class",axis=1).columns
X=df.drop("Class",axis=1)
y=df["Class"]
scaler = StandardScaler()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=7,stratify=y)
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
smote=SMOTE(random_state=42)
X_train,y_train=smote.fit_resample(X_train,y_train)
models={"Logistic Regression":LogisticRegression(max_iter=1000,random_state=42,class_weight="balanced"),
        "Random Forest":RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,class_weight="balanced",n_jobs=-1),}
best_model=""
best_acc=0
results=[]
for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)[:,1]
    acc=accuracy_score(y_test,y_pred)
    print("=" * 60)
    print(name)
    print("=" * 60)
    print("Precision:",precision_score(y_test,y_pred))
    print("Recall:",recall_score(y_test,y_pred))
    print("Classification Report:",classification_report(y_test,y_pred))
    print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
    print("ROC AUC:",roc_auc_score(y_test,y_prob))
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob)
    })
    if acc>best_acc:
        best_model=name
        best_acc=acc
    fpr, tpr, threshold = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, linewidth=2, label=name)
plt.plot([0,1], [0,1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.savefig("images/ROC_curve.png")
plt.close()
plt.show()
results_df = pd.DataFrame(results)
print("\nModel Comparison")
print(results_df)
joblib.dump(models[best_model], "best_model.pkl")
print("Best model saved successfully.")
print("\nBest Model :", best_model)
print("Best Accuracy :", best_acc)
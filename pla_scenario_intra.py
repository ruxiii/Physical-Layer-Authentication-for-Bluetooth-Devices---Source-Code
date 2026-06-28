import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# citire dataset inter-brand
df = pd.read_csv(r"D:\Users\ruxiii\OneDrive\Desktop\disertatie\rezultate\intraband_dataset.csv")

# alegem dispozitivul legitim
legitimate_device = "iphone6"

# reformulare in scenariu de autentificare
df["auth_label"] = df["label"].apply(
    lambda x: "legitimate" if x == legitimate_device else "attacker"
)

# features si etichete
X = df.drop(columns=["label", "file_name", "auth_label"])
y = df["auth_label"]

# impartire train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# predictii
y_pred = clf.predict(X_test)

# evaluare 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# confusion matrix
labels = ["legitimate", "attacker"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("\nConfusion matrix:")
print(cm)

# FAR si FRR
TP = cm[0, 0]
FN = cm[0, 1]
FP = cm[1, 0]
TN = cm[1, 1]

FRR = FN / (TP + FN) if (TP + FN) > 0 else 0.0
FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0

print(f"\nFRR: {FRR:.4f}")
print(f"FAR: {FAR:.4f}")
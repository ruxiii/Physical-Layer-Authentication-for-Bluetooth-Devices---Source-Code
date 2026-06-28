import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# citire dataset inter-brand
df = pd.read_csv(r"D:\Users\ruxiii\OneDrive\Desktop\disertatie\rezultate\interbrand_dataset.csv")

# features si etichete
X = df.drop(columns=["label", "file_name"])
y = df["label"]

# impartire train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# definire modele
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=3)
}

results = []

best_model_name = None
best_y_pred = None
best_accuracy = -1

# antrenare si evaluare
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Macro F1": macro_f1
    })

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("Macro F1:", macro_f1)
    print(classification_report(y_test, y_pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_y_pred = y_pred

# rezultate
results_df = pd.DataFrame(results)

print("\nFinal results:")
print(results_df)

# salvare rezultate in csv
results_df.to_csv(r"D:\Users\ruxiii\OneDrive\Desktop\disertatie\rezultate\interbrand_results.csv", index=False)

# afisare cel mai bun model si matricea de confuzie
print(f"\nBest model: {best_model_name}")
print("Confusion matrix:")
print(confusion_matrix(y_test, best_y_pred))
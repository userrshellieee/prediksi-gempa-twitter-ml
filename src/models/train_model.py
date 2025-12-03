import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =============== 1. BACA DATASET ===============
print("[INFO] Membaca dataset_final.csv dari folder ini ...")
df = pd.read_csv("dataset_final.csv")

if "text_clean" not in df.columns or "target" not in df.columns:
    raise Exception("dataset_final.csv harus punya kolom 'text_clean' dan 'target'")

# buang baris kosong dan pastikan tipe datanya benar
df = df.dropna(subset=["text_clean", "target"])
X_text = df["text_clean"].astype(str)
y = df["target"].astype(int)

# =============== 2. TRAIN / TEST SPLIT ===============
print("[INFO] Membagi data train / test ...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# =============== 3. TF-IDF VECTORIZER ===============
print("[INFO] Mengubah teks menjadi vektor TF-IDF ...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# =============== 4. LOGISTIC REGRESSION ===============
print("[INFO] Training model Logistic Regression ...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =============== 5. EVALUASI ===============
print("[INFO] Evaluasi model ...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n=== AKURASI ===")
print(acc)

print("\n=== CLASSIFICATION REPORT ===")
cls_report = classification_report(y_test, y_pred)
print(cls_report)

cm = confusion_matrix(y_test, y_pred)
print("\n=== CONFUSION MATRIX ===")
print(cm)

# =============== 6. SIMPAN CONFUSION MATRIX ===============
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-disaster (0)", "Disaster (1)"],
    yticklabels=["Non-disaster (0)", "Disaster (1)"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("[DONE] confusion_matrix.png disimpan di folder ini.")

# =============== 7. SIMPAN MODEL & VECTORIZER ===============
joblib.dump(model, "logreg_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("[DONE] logreg_model.pkl dan tfidf_vectorizer.pkl disimpan di folder ini.")
print("\n[SELESAI] Training Logistic Regression selesai.")

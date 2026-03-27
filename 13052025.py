
!pip install -U sentence-transformers transformers scikit-learn


from google.colab import drive
drive.mount('/content/drive')


import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gc


df = pd.read_csv('/content/drive/MyDrive/tayyab/italian_crime_news.csv')
texts = df['text'].tolist()
labels = df['word2vec_tag'].tolist()
del df
gc.collect()


X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
del texts, labels
gc.collect()


model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


print("Encoding training data...")
emb_train = model.encode(X_train, convert_to_numpy=True, show_progress_bar=True)
print("Encoding test data...")
emb_test  = model.encode(X_test,  convert_to_numpy=True, show_progress_bar=True)


del X_train, X_test
gc.collect()


classifiers = {
    'SVM (linear)':        SVC(kernel='linear', probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42)
}

def evaluate(clf, X_tr, y_tr, X_te, y_te):
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, preds, average='weighted', zero_division=0
    )
    return acc, prec, rec, f1


results = []
for name, clf in classifiers.items():
    print(f"Training & evaluating: {name}")
    acc, prec, rec, f1 = evaluate(clf, emb_train, y_train, emb_test, y_test)
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1
    })


results_df = pd.DataFrame(results)
print("\n=== Classification Results ===")
print(results_df)


results_df.to_csv('/content/drive/MyDrive/tayyab/classifier_comparison_results.csv', index=False)
print("\nSaved comparison to Drive.")

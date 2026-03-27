

!pip install -U sentence-transformers transformers scikit-learn tqdm


from google.colab import drive
drive.mount('/content/drive')


import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


DATA_PATH = '/content/drive/MyDrive/Tayyab/italian_crime_news.csv'
df = pd.read_csv(DATA_PATH)


# 5. Train-test split
texts = df['text'].astype(str).tolist()
labels = df['word2vec_tag'].tolist()
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=labels)
X_train = df_train['text'].astype(str).tolist()
X_test  = df_test['text'].astype(str).tolist()
y_train = df_train['word2vec_tag'].tolist()
y_test  = df_test['word2vec_tag'].tolist()


print("Loading Italian BERT model...")
tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')
model     = AutoModel.from_pretrained('dbmdz/bert-base-italian-cased')
model.eval()
for p in model.parameters():
    p.requires_grad = False


def get_batch_embeddings(texts, tokenizer, model, batch_size=16, device='cpu'):
    all_embs = []
    model.to(device)
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        
        batch_texts = [str(t) for t in batch]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  
        mask = inputs['attention_mask'].unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = (summed / counts).cpu().numpy()
        all_embs.append(pooled)
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    return np.vstack(all_embs)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
emb_train = get_batch_embeddings(X_train, tokenizer, model, batch_size=256, device=device)
emb_test  = get_batch_embeddings(X_test,  tokenizer, model, batch_size=256, device=device)


def evaluate_models(train_emb, test_emb, y_train, y_test):
    results = []
    classifiers = {
        'SVM (linear)': SVC(kernel='linear', probability=True),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Gradient Boosting': HistGradientBoostingClassifier()
    }
    for name, clf in classifiers.items():
        print(f"Training and evaluating: {name}")
        clf.fit(train_emb, y_train)
        preds = clf.predict(test_emb)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1
        })
    return pd.DataFrame(results)


results_df = evaluate_models(emb_train, emb_test, y_train, y_test)
print("\nEvaluation Results:\n", results_df)

results_df.to_csv('/content/drive/MyDrive/Tayyab/classifier_comparison_results2.csv', index=False)
print("\nSaved comparison to Drive.")
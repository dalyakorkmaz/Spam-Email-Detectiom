import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Dataset'i oku
data = pd.read_csv("spam.csv", encoding="latin-1")

# 2. Sadece gerekli sütunları al
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# 3. Label'ı sayıya çevir
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# 4. Girdi ve çıktı
X = data['message']
y = data['label_num']

# 5. Eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Metinleri sayısal hale getir
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Model oluştur ve eğit
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 8. Tahmin yap
y_pred = model.predict(X_test_vec)

# 9. Sonuçları yazdır
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 10. Elle test
new_messages = [
    "Congratulations! You have won a free ticket. Call now!",
    "Hey, are we meeting after class today?"
]

new_messages_vec = vectorizer.transform(new_messages)
predictions = model.predict(new_messages_vec)

print("\nNew Message Predictions:")
for msg, pred in zip(new_messages, predictions):
    if pred == 1:
        print(f"SPAM: {msg}")
    else:
        print(f"HAM: {msg}")
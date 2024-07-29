import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Question 10: The result of the program that reads a file and uses TF-IDF to represent the text into the vector:
vi_data_df = pd.read_csv('./Module_2/Week_4/Data/vi_text_retrieval.csv')
context = vi_data_df['text']
context = [doc.lower() for doc in context]

tfidf_vectorizer = TfidfVectorizer()
context_embedded = tfidf_vectorizer.fit_transform(context)
print(round(context_embedded.toarray()[7][0], 2)) # 0.31

# Question 11: The result of the cosine similarity calculation program is:
def tfidf_search(question, tfidf_vectorizer, top_d=5):
    # lowercasing before encoding
    query_embedded = tfidf_vectorizer.transform([question.lower()])
    cosine_scores = cosine_similarity(query_embedded, context_embedded).reshape((-1,))
    
    # Get top k cosine score and index its
    results = []
    for idx in cosine_scores.argsort()[-top_d:][::-1]:
        doc_score = {
            'id': idx,
            'cosine_score': cosine_scores[idx]
        }
        results.append(doc_score)
    return results

question = vi_data_df.iloc[0]['question']
results = tfidf_search(question, tfidf_vectorizer, top_d=5)
print(round(results[0]['cosine_score'], 2)) # 0.63

# Question 12: The result of the correlation similarity calculation program is:
def corr_search(question, tfidf_vectorizer, top_d=5):
    # lowercasing before encoding
    query_embedded = tfidf_vectorizer.transform([question.lower()])
    corr_scores = np.corrcoef(query_embedded.toarray()[0], context_embedded.toarray())
    corr_scores = corr_scores[0][1:]
    
    # Get top k cosine score and index its
    results = []
    for idx in corr_scores.argsort()[-top_d:][::-1]:
        doc_score = {
            'id': idx,
            'corr_score': corr_scores[idx]
        }
        results.append(doc_score)
    return results

question = vi_data_df.iloc[0]['question']
results = corr_search(question, tfidf_vectorizer, top_d=5)
print(round(results[1]['corr_score'], 2)) # 0.21
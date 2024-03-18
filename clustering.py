#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:49:59 2024

@author: diego
"""

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

#This downloads the Punkt tokenizer models
nltk.download('punkt')
#stop words
nltk.download('stopwords')
#WordNet, Large lexical database of English. It is used for lemmatization (running -> run)
nltk.download('wordnet')

# Directory containing text files
directory = "/home/diego/Documents/Australia/Qstudio/articles"

# Load and preprocess the text files
texts = []
file_names=[]
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        with open(os.path.join(directory, filename), 'r') as f:
            text = f.read()
            # Tokenize, remove stop words, and lemmatize
            tokens = word_tokenize(text)
            #word.isalpha() take just alphabetic characters
            tokens = [WordNetLemmatizer().lemmatize(word.lower()) for word in tokens if word.isalpha()]
            tokens = [word for word in tokens if word not in stopwords.words('english')]
            texts.append(' '.join(tokens))
            file_names.append(filename)

# Feature extraction
#Term Frequency-Inverse Document Frequency
#measure used to evaluate the importance of a word to a document in a corpus
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)



# Find the best number of clusters
silhouette_scores = []
K = range(2, 10)  

for k in K:
    # assign each observation to the cluster with the nearest mean or centroid based on its features
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    cluster_labels = kmeanModel.predict(X)
    #metric used to assess the quality of clusters created by a clustering algorithm
    #measures how similar an object is to its own cluster compared to other clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plotting results of silhouette_score
plt.figure(figsize=(16,8))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score showing the optimal number of clusters')
plt.grid()
plt.show()


# Clustering with the best number of clusters
num_clusters = 6
km = KMeans(n_clusters=num_clusters)
km.fit(X)

# Assign documents to clusters
clusters = km.labels_.tolist()

# Display cluster assignment
for file, cluster in zip(file_names,clusters):
    print(f"{file} is in cluster {cluster}")
    

#Visualize dataset in 2D
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X.toarray())

# Create a scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_X[:, 0], reduced_X[:, 1], c=clusters, cmap='viridis')
plt.title('Cluster assignments')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


for i, txt in enumerate(file_names):
    plt.annotate(txt.replace('.txt', ''), (reduced_X[i, 0], reduced_X[i, 1]))


plt.legend(handles=scatter.legend_elements()[0], labels=set(clusters))

plt.show()



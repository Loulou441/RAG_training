# RAG Hybride avec Groq & Sentence-Transformers ⚡

Ce projet met en place un pipeline **RAG (Retrieval-Augmented Generation)** complet, modulaire et ultra-rapide en combinant la puissance de calcul locale pour les embeddings sémantiques et la vitesse de l'architecture LPU de **Groq** pour la génération textuelle.

L'objectif est d'implémenter l'ensemble du processus à des fins d'apprentissage, sans dépendre de frameworks d'orchestration abstraits (comme LangChain ou LlamaIndex).

## ⚙️ Architecture du Pipeline Hybride
Groq se concentrant exclusivement sur l'inférence ultra-rapide de LLM de pointe, l'API ne fournit pas de modèle de vectorisation. Ce projet adopte donc une architecture hybride standard de l'industrie :
1. **Embeddings Locaux (Ingestion)** : Utilisation de `sentence-transformers` avec le modèle `all-MiniLM-L6-v2` pour encoder les documents textuels en vecteurs numériques de manière gratuite et locale.
2. **Recherche Sémantique (Retrieval)** : Calcul manuel de la **similarité cosinus** (`cosine_similarity`) en pur Python pour comparer l'angle vectoriel entre la question de l'utilisateur et les documents stockés.
3. **Génération Contextuelle (Augmentation)** : Injection sélective des meilleurs documents correspondants dans un prompt direct et exécution de la génération via l'API **Groq** avec le modèle `openai/gpt-oss-20b`.

## 🛠️ Structure du Projet
```text
rag_training/
│
├── src
    ├──app.py           # Logique et calculs fondamentaux du pipeline RAG
    ├── config.py       # Configuration du projet
├── .env.exemple        # exemple du .env à complété
├── requirements.txt    # Dépendances requises
└── README.md           # Documentation technique
import os
import math
from groq import Groq
from sentence_transformers import SentenceTransformer
from config import GROQ_API_KEY, SENTENCE_TRANSFORMERS_MODEL, LLM_MODEL

# 1. Initialisation du client Groq et du modèle d'embeddings
# Assurez-vous d'avoir configuré votre clé API : export GROQ_API_KEY="gsk_..."
client = Groq(api_key=GROQ_API_KEY)

# On utilise un modèle d'embedding léger et très performant en français, exécuté en local
embedding_model = SentenceTransformer(SENTENCE_TRANSFORMERS_MODEL)

# 2. Base de connaissances locale (Simule notre base de documents)
DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "Procédure Sinistre Auto Allianz",
        "text": "En cas de sinistre automobile, l'assuré doit remplir un constat amiable et le transmettre dans les 5 jours ouvrés. Pour les dommages corporels, une expertise médicale est automatiquement déclenchée sous 48 heures pour évaluer les préjudices."
    },
    {
        "id": "doc_2",
        "title": "Garanties Contrat Corporel",
        "text": "La garantie Accidents de la Vie couvre les dommages corporels jusqu'à hauteur de 1 000 000 d'euros en cas d'incapacité permanente supérieure ou égale à 10%. Les frais médicaux restants à charge sont remboursés après intervention de la Sécurité Sociale."
    },
    {
        "id": "doc_3",
        "title": "Exclusions Auto Standard",
        "text": "Sont exclues des garanties auto standard les pannes mécaniques liées à l'usure normale du véhicule, ainsi que les sinistres survenus lorsque le conducteur est en état d'ébriété ou sans permis de conduire valide."
    }
]

def get_embedding(text: str) -> list:
    """Génère un vecteur d'embedding localement sans appel API."""
    vector = embedding_model.encode(text)
    return vector.tolist()

def cosine_similarity(v1: list, v2: list) -> float:
    """Calcule la similarité cosinus entre deux vecteurs."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def build_knowledge_base():
    """Génère les embeddings pour tous nos documents de référence."""
    print("--- Ingestion : Génération des embeddings (en local via SentenceTransformers) ---")
    for doc in DOCUMENTS:
        print(f"Indexation du document: {doc['title']}...")
        doc['embedding'] = get_embedding(doc['text'])
    print("Base de connaissances prête !\n")

def retrieve(query: str, top_k: int = 1) -> list:
    """Recherche les documents les plus pertinents par rapport à la question."""
    print(f"--- Requête : '{query}' ---")
    query_embedding = get_embedding(query)
    
    scored_docs = []
    for doc in DOCUMENTS:
        similarity = cosine_similarity(query_embedding, doc['embedding'])
        scored_docs.append((similarity, doc))
    
    # Tri par score de similarité décroissant
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return scored_docs[:top_k]

def generate_answer(query: str, retrieved_docs: list) -> str:
    """Augmente le prompt et appelle Groq pour une génération ultra-rapide."""
    print("--- Génération : Construction du prompt augmenté et appel à l'API Groq ---")
    
    context_str = ""
    for score, doc in retrieved_docs:
        context_str += f"\nSource: {doc['title']}\nContenu: {doc['text']}\n(Score de similarité: {score:.4f})\n"
    
    prompt = f"""Tu es un assistant IA expert en gestion des sinistres et contrats d'assurance.
Utilise UNIQUEMENT les informations du contexte fourni ci-dessous pour répondre à la question de l'utilisateur. 
Si la réponse ne se trouve pas dans le contexte, dis poliment que tu ne disposes pas de cette information.

CONTEXTE DE RÉFÉRENCE :
{context_str}

QUESTION DE L'UTILISATEUR :
{query}

RÉPONSE PRÉCISE ET SOURCÉE :"""

    # Appel au modèle Llama 3 sur Groq (Vitesse d'inférence maximale)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=LLM_MODEL,
    )
    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    # Vérification de la clé API Groq
    if not os.environ.get("GROQ_API_KEY"):
        print("Erreur : La variable d'environnement GROQ_API_KEY n'est pas configurée.")
        print("Veuillez l'ajouter avec : export GROQ_API_KEY='gsk_...'")
        exit(1)
        
    # Phase 1 : Ingestion des documents
    build_knowledge_base()
    
    # Phase 2 : Question/Réponse (Exemple 1)
    user_query_1 = "Que se passe-t-il si j'ai un accident corporel et combien de temps ai-je pour envoyer mon constat ?"
    results = retrieve(user_query_1, top_k=2)
    answer = generate_answer(user_query_1, results)
    
    print("\n=== RÉPONSE DE GROQ ===")
    print(answer)
    print("=======================\n")
    
    # Exemple 2 : Hors contexte
    user_query_2 = "Quelle est la capitale de la France ?"
    results_2 = retrieve(user_query_2, top_k=1)
    answer_2 = generate_answer(user_query_2, results_2)
    
    print("\n=== RÉPONSE DE GROQ (HORS CONTEXTE) ===")
    print(answer_2)
    print("=======================")
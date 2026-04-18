import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Détecteur de Spam", page_icon="📧", layout="centered")

# --- CHARGEMENT DES RESSOURCES (Mise en cache pour la rapidité) ---
@st.cache_resource
def load_models():
    # Ajuste les chemins si nécessaire
    vec = joblib.load('modeles/tfidf_vectorizer.pkl')
    mod = joblib.load('modeles/modele_spam_choisi.pkl')
    return vec, mod

vectorizer, model = load_models()

# Préparation NLP
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# Au cas où punkt_tab serait requis par ton environnement
nltk.download('punkt_tab', quiet=True) 
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Fonction de nettoyage du texte."""
    text = str(text).lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- INTERFACE UTILISATEUR ---
st.title("📧 Détecteur de Spam IA")
st.write("Remplissez le sujet et/ou le corps de l'email ci-dessous. Notre modèle **Random Forest** analysera le texte fusionné pour déterminer s'il s'agit d'un Spam ou d'un email légitime.")

# Zones de saisie séparées
sujet_input = st.text_input("Sujet de l'email :", placeholder="Ex: URGENT! You have won a $1000 gift card")
message_input = st.text_area("Corps de l'email :", height=200, placeholder="Ex: Click the link below to claim your free money now!...")

# Bouton d'action
if st.button("Analyser l'email", type="primary"):
    # On vérifie si les DEUX champs sont vides
    if sujet_input.strip() == "" and message_input.strip() == "":
        st.warning("⚠️ Veuillez entrer au moins un sujet ou le corps d'un message avant de lancer l'analyse.")
    else:
        with st.spinner("Analyse en cours..."):
            # 0. Fusionner le sujet et le message (comme lors de l'entraînement)
            texte_fusionne = f"{sujet_input} {message_input}".strip()
            
            # 1. Nettoyage
            cleaned_text = preprocess_text(texte_fusionne)
            
            # 2. Vectorisation
            vectorized_text = vectorizer.transform([cleaned_text])
            
            # 3. Prédiction
            prediction = model.predict(vectorized_text)[0]
            
            # 4. Affichage du résultat
            st.markdown("---")
            if prediction == 1:
                st.error("🚨 **ALERTE SPAM** : Cet email est considéré comme indésirable.")
            else:
                st.success("✅ **EMAIL NORMAL (HAM)** : Cet email semble légitime.")
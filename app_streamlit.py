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
st.write("Entrez le texte d'un email ci-dessous. Notre modèle **Random Forest** analysera son contenu pour déterminer s'il s'agit d'un Spam ou d'un email légitime.")

# Zone de texte
email_input = st.text_area("Collez l'email ici :", height=200, placeholder="Ex: URGENT! You have won a $1000 gift card...")

# Bouton d'action
if st.button("Analyser l'email", type="primary"):
    if email_input.strip() == "":
        st.warning("⚠️ Veuillez entrer du texte avant de lancer l'analyse.")
    else:
        with st.spinner("Analyse en cours..."):
            # 1. Nettoyage
            cleaned_text = preprocess_text(email_input)
            
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
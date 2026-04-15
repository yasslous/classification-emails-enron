from flask import Flask, request, jsonify
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du Vectorizer et du Modèle Random Forest depuis le dossier 'modeles'
# Ajuste les noms des fichiers si tu les as nommés différemment
vectorizer = joblib.load('modeles/tfidf_vectorizer.pkl')
model = joblib.load('modeles/modele_spam_choisi.pkl')

# Chargement des stopwords (nécessaire pour la fonction de nettoyage)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Même fonction de nettoyage que dans le notebook."""
    text = str(text).lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Création de la route (le point d'entrée de l'API)
@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données envoyées en JSON
    data = request.get_json()
    
    # Vérifier que l'utilisateur a bien envoyé un champ 'email'
    if not data or 'email' not in data:
        return jsonify({'error': "Veuillez fournir un texte dans le champ 'email'."}), 400
    
    # 1. Récupération du texte brut
    email_text = data['email']
    
    # 2. Nettoyage du texte
    cleaned_text = preprocess_text(email_text)
    
    # 3. Vectorisation (Transformation en nombres)
    # Attention : on utilise bien transform() ici, pas fit_transform()
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # 4. Prédiction avec le Random Forest
    prediction = model.predict(vectorized_text)[0]
    
    # 5. Interprétation du résultat (1 = Spam, 0 = Ham)
    resultat = "Spam" if prediction == 1 else "Ham"
    
    # Renvoi de la réponse en format JSON
    return jsonify({
        'email_analyse': email_text,
        'prediction': resultat,
        'code_prediction': int(prediction)
    })

if __name__ == '__main__':
    # Lancement du serveur en mode développeur
    app.run(debug=True, port=5000)
📧 Détecteur de Spam IA : Classification d'Emails avec Machine Learning et NLP

🌟 Aperçu et Motivation

À l'ère du numérique, l'email est devenu un moyen de communication critique dans tous les domaines (personnel, professionnel, gouvernemental). Cependant, cette utilisation massive s'accompagne d'un fléau : le spam. À titre d'exemple, une étude a révélé qu'en 2020, sur 300 milliards d'emails envoyés quotidiennement, 170 milliards étaient identifiés comme des spams.


Détecter ces messages indésirables est un défi complexe en Traitement du Langage Naturel (NLP) car les spams sont souvent des textes courts, trompeurs et conçus spécifiquement pour échapper aux systèmes de détection. Ce projet a pour but de relever ce défi en construisant un système d'intelligence artificielle capable d'analyser le contenu textuel des emails (sujet et message) pour les classifier automatiquement comme "Spam" ou "Ham" (email légitime).


🔬 Inspiration et Comparaison avec la Littérature Scientifique

Ce projet s'inspire des récents travaux de recherche, notamment l'article "Spam Classification Using Machine Learning: A Survey" (2025). Cet article souligne un paradoxe majeur : bien que les méthodes de Machine Learning aient un potentiel énorme, elles sont rarement implémentées dans des environnements de production réels.

Mon travail vise précisément à combler ce fossé entre la théorie et la pratique. Voici comment mon approche se compare et s'aligne avec la recherche actuelle :


Le choix du jeu de données : Tout comme les chercheurs qui ont utilisé le célèbre dataset "Enron1" pour développer des modèles de classification robustes , j'ai basé mon entraînement sur les données d'Enron pour garantir une diversité linguistique pertinente.


L'excellence des modèles classiques : L'étude confirme que des méthodes basées sur la fréquence des mots couplées à des algorithmes comme Naive Bayes et Random Forest sont hautement robustes. Dans mon projet, après optimisation des hyperparamètres (GridSearchCV), j'ai pu démontrer que ces deux modèles excellent pour cette tâche, frôlant la perfection avec très peu de surapprentissage.


Techniques NLP et Extraction de caractéristiques : La recherche explore des méthodes d'extraction complexes comme l'ADF-CTF (Advanced Deep Feature-Contextual Term Frequency) pour extraire les caractéristiques les plus significatives. Dans mon approche, j'ai implémenté la technique classique et redoutable TF-IDF (Term Frequency-Inverse Document Frequency), accompagnée d'un prétraitement NLP rigoureux (tokenisation et suppression des mots vides). Cette stratégie a prouvé qu'une extraction de caractéristiques bien maîtrisée suffit à obtenir des performances de pointe.


📂 Structure du Projet

Le projet a été développé de manière itérative à travers plusieurs Notebooks Jupyter, documentant chaque étape du cycle de vie de la donnée :

1. 01_exploration_donnees.ipynb
Objectif : Comprendre la répartition et la nature des données.

Actions : Analyse exploratoire du dataset Enron, vérification de l'équilibre des classes (Spam vs Ham), et Vérification les tailles des différents emails "spam/Ham" via des visuels.J'ai découvert que les spams ont de taille courte par rapport les Hams.

2. 02_pretraitement_nlp.ipynb
Objectif : Préparer le texte brut pour la machine.

Actions : Implémentation des techniques de NLP. Les textes ont été convertis en minuscules, la ponctuation a été supprimée, puis nous avons appliqué la Tokenisation (découpage en mots) , la suppression des "Stop Words" (mots vides qui n'aident pas à la classification) et et création de visualisation comme  les WordCloud pour identifier les mots les plus fréquents utilisés par les spammeurs par rapport aux emails normaux. Les données nettoyées ont été sauvegardées pour l'entraînement.


3. 03_entrainement_modeles.ipynb
Objectif : Créer l'intelligence du système.

Actions : Application de la vectorisation TF-IDF pour transformer le texte en données numériques. Entraînement et comparaison de deux modèles : Naive Bayes et Random Forest. Nous avons évalué les modèles à l'aide de matrices de confusion, de l'Accuracy, Precision, Recall et F1-Score. Enfin, nous avons utilisé la validation croisée et GridSearchCV pour optimiser les hyperparamètres et prouver l'excellence de ces algorithmes sans tomber dans le surapprentissage.

🚀 Déploiement : L'Application Streamlit
Le modèle Random Forest s'étant démarqué par ses résultats légèrement supérieurs et sa grande robustesse, il a été sélectionné comme modèle final.

Pour rendre ce modèle accessible et utilisable (répondant ainsi au défi de l'implémentation en environnement réel soulevé par les chercheurs ), j'ai développé une application web interactive avec Streamlit (app_streamlit.py).

Fonctionnalité : L'interface permet à n'importe quel utilisateur de saisir le "Sujet" et/ou le "Corps" d'un email suspect.

Traitement en temps réel : L'application applique instantanément le même pipeline NLP (nettoyage, tokenisation, TF-IDF) que lors de l'entraînement, puis interroge le modèle Random Forest sauvegardé pour prédire en temps réel s'il s'agit d'un Spam (alerte rouge) ou d'un email légitime (Ham, succès vert).

*** Outils et Bibliothèques utilisés : Python, Pandas, Scikit-Learn (TF-IDF, Random Forest, Naive Bayes), NLTK (Natural Language Toolkit), Matplotlib/Seaborn (WordCloud), Streamlit.

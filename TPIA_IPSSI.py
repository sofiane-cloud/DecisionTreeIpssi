# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Charger les données
df=pd.read_csv('car_evaluation.csv')
# Afficher les premières lignes pour vérifier les données
print(df.head())

df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'CAR']
print(df.head())
# Encodage des variables catégorielles
# On utilise LabelEncoder pour transformer les valeurs textuelles en valeurs numériques
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Définir les caractéristiques (features) et la cible (target)
X = df[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]  # Attributs
y = df['CAR']  # Label

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer et entraîner l'arbre de décision
clf = DecisionTreeClassifier(max_depth=4, random_state=42)  # Limiter la profondeur pour éviter la surapprentissage
clf.fit(X_train, y_train)

# Afficher les performances sur l'ensemble de test
accuracy = clf.score(X_test, y_test)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

# Visualisation de l'arbre
plt.figure(figsize=(16, 10))
plot_tree(
    clf,
    feature_names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
    class_names=label_encoders['CAR'].classes_,
    filled=True
)
plt.title("Arbre de Décision pour la Classification des Voitures", fontsize=16)
plt.show()

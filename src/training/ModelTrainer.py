# Pipeline verbindet Preprocessing und Modelltraining
from sklearn.pipeline import Pipeline

# Logistische Regression für Klassifikationsprobleme
from sklearn.linear_model import LogisticRegression

# Decision Tree für regelbasierte Klassifikation
from sklearn.tree import DecisionTreeClassifier

# Klasse zum Trainieren der Machine-Learning-Modelle
class ModelTrainer:
    
    # Konstruktor der Klasse
    # Speichert den Preprocessor
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor


    # Diese Methode erstellt und trainiert
    # die logistische Regression
    def train_logistic_regression(self, X_train, y_train):

        # Pipeline verbindet die Datenvorbereitung und das ModelltrainingModelltraining
        #Dadurch werden alle Schritte automatisch ausgeführt.

        logistic_pipeline = Pipeline(steps=[

            # Führt Encoding und Skalierung aus
            ("preprocessor", self.preprocessor),

            # Erstellt das Modell
            (
                "classifier",

                # max_iter=1000 erhöht die maximale Anzahlder Trainingsdurchläufe.
                # Dadurch wird verhindert,dass das Training zu früh abbricht.
                LogisticRegression(max_iter=1000)
            )
        ])

        # Modell mit Trainingsdaten trainieren
        logistic_pipeline.fit(X_train, y_train)

        print("\nLogistische Regression wurde trainiert.")

        return logistic_pipeline


    # Diese Methode erstellt und trainiertden Decision Tree
    def train_decision_tree(self, X_train, y_train):

        # Pipeline für den Decision Tree
        decision_tree_pipeline = Pipeline(steps=[
            # Führt Preprocessing aus
            ("preprocessor", self.preprocessor),

            # Erstellt den Decision Tree
            (
                "classifier",

                # max_depth=4 begrenzt die Tiefe des Baums.
                # Dadurch wird Overfitting reduziert.
                # random_state sorgt für reproduzierbare Ergebnisse.
                DecisionTreeClassifier(
                    max_depth=4,
                    random_state=42
                )
            )
        ])

        # Modell trainieren
        decision_tree_pipeline.fit(X_train, y_train)

        print("\nDecision Tree wurde trainiert.")

        return decision_tree_pipeline
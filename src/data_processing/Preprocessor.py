# ColumnTransformer ermöglicht unterschiedliche Verarbeitungsschritte
# für verschiedene Spalten des Datensatzes
from sklearn.compose import ColumnTransformer

# OneHotEncoder wandelt Textwerte in numerische Werte um
# StandardScaler skaliert numerische Werte für stabileres Modelltraining
from sklearn.preprocessing import OneHotEncoder, StandardScaler



# Klasse zur Vorbereitung der Daten für Machine Learning
class Preprocessor:


    # Konstruktor der Klasse
    def __init__(self):

        # Kategoriale Features enthalten Textwerte
        # Machine-Learning-Modelle können Text nicht direkt verarbeiten.
        # Deshalb müssen diese Spalten später kodiert werden.
        self.categorical_features = ["color", "size"]

        # Numerische Features enthalten Zahlenwerte
        # Diese Werte können direkt verwendet werden, sollten aber skaliert werden.
        self.numeric_features = ["weight"]


    # Diese Methode erstellt den Preprocessor
    def create_preprocessor(self):

        # ColumnTransformer ermöglicht unterschiedliche Verarbeitungsschritte für verschiedene Spalten.
        # Dadurch können Textspalten kodiert und Zahlenwerte skaliert werden.
        preprocessor = ColumnTransformer(
            transformers=[

                # Verarbeitung der kategorialen Features
                (
                    "cat",

                    # OneHotEncoder wandelt Textwerte in numerische Werte um.
                    # handle_unknown="ignore" verhindert Fehler, falls später unbekannte Kategorien auftreten.
                    OneHotEncoder(handle_unknown="ignore"),

                    # Spalten, auf die der Encoder angewendet wird
                    self.categorical_features
                ),

                # Verarbeitung numerischer Features
                (
                    "num",

                    # StandardScaler skaliert numerische Werte.
                    # Dadurch liegen große und kleine Werte in einem ähnlichen Wertebereich.
                    #
                    # Das verbessert Stabilität, Trainingsqualität, Modellleistung
                    StandardScaler(),

                    # Spalten, auf die die Skalierung angewendet wird
                    self.numeric_features
                )
            ]
        )

        # Kontrollausgabe
        print("\nPreprocessor wurde erstellt.")

        # Gibt den fertigen Preprocessor zurück
        return preprocessor

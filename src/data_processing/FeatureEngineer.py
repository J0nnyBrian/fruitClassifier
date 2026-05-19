# Klasse zur Vorbereitung der Machine-Learning-Daten
class FeatureEngineer:

    # Konstruktor der Klasse
    # Speichert den Datensatz
    def __init__(self, dataframe):
        self.df = dataframe

    # Diese Methode erstellt die Eingabedaten für das Modell
    def create_features(self):

        # Auswahl der relevanten Merkmale
        # Diese Features beschreiben die Eigenschaften der Früchte
        X = self.df[["color", "size", "weight"]]

        print("\nFeatures wurden erstellt.")

        return X

    # Diese Methode erstellt die Zielvariable
    def create_target(self):

        # Die Zielvariable enthält die Fruchtarten,
        # die das Modell später vorhersagen soll
        y = self.df["fruit_type"]

        print("\nZielvariable wurde erstellt.")

        return y
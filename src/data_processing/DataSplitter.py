from sklearn.model_selection import train_test_split

# Klasse zum Aufteilen der Daten in Trainings- und Testdaten
class DataSplitter:

    # Konstruktor der Klasse
    # Speichert Features und Zielvariable
    def __init__(self, X, y):
        self.X = X
        self.y = y

   
   # Diese Methode teilt die Daten in Trainings- und Testdaten auf
    def split_data(self):

        X_train, X_test, y_train, y_test = train_test_split(
        # train_test_split() teilt die Daten zufällig in Trainings- und Testdaten auf.
        #
        # Trainingsdaten werden verwendet, damit das Modell Muster lernt.
        # Testdaten werden verwendet, um die Modellleistung später zu bewerten.
        #
        # - 75% der Daten werden zum Trainieren verwendet
        # - 25% der Daten werden zum Testen verwendet
        #
        # random_state=42 sorgt dafür,
        # dass die Datenaufteilung bei jedem Programmstart gleich bleibt.
        # Dadurch bleiben Ergebnisse reproduzierbar und nachvollziehbar.
        #
        # stratify=self.y sorgt dafür,
        # dass alle Fruchtarten gleichmäßig in Trainings- und Testdaten verteilt werden.
        # Dadurch wird verhindert, dass bestimmte Klassen im Testdatensatz fehlen.
        self.X,
        self.y,
        test_size=0.25,
        random_state=42,
        stratify=self.y
)
        print("\nDaten wurden in Trainings- und Testdaten aufgeteilt.")

        return X_train, X_test, y_train, y_test
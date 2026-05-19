# pandas importieren -> Bibliothek für Datenanalyse und Tabellenverarbeitung
import pandas as pd

# Importiert die Klasse zur Datenbereinigung
from data_processing.DataCleaner import DataCleaner

# Importiert die Klasse zur Vorbereitung der Machine-Learning-Daten
from data_processing.FeatureEngineer import FeatureEngineer

# Importiert die Klasse zum Aufteilen der Daten
from data_processing.DataSplitter import DataSplitter

# Importiert die Klasse für Encoding und Skalierung
from data_processing.Preprocessor import Preprocessor

# Importiert die Klasse zum Trainieren der Modelle
from training.ModelTrainer import ModelTrainer

# Importiert die Klasse zur Bewertung der Machine-Learning-Modelle
from evaluation.ModelEvaluator import ModelEvaluator

# joblib speichert trainierte Machine-Learning-Modelle
import joblib

# Excel-Datei laden
# read_excel() liest die Excel-Datei ein
# df steht für "DataFrame"(tabellarische Datenstruktur)
df = pd.read_excel("data/fruit_data.xlsx")

# Erste Zeilen anzeigen
# head() zeigt standardmäßig die ersten 5 Zeilen des Datensatzes
print("Erste 5 Zeilen:")
print(df.head())

# Allgemeine Informationen
# info() zeigt:
# - Anzahl der Zeilen
# - Datentypen
# - fehlende Werte
# - Speicherverbrauch
print("\nInformationen zum Datensatz:")
print(df.info())

# Fehlende Werte prüfen
# isna() erkennt fehlende Werte (NaN)
# sum() zählt die Anzahl fehlender Werte pro Spalte
print("\nFehlende Werte:")
print(df.isna().sum())

# Statistische Übersicht
# describe() berechnet Mittelwert, Minimum, Maximum, Standardabweichung usw.
print("\nStatistische Übersicht:")
print(df.describe())

# Kategorien prüfen
# unique() zeigt alle unterschiedlichen Werte einer Spalte
print("\nFruchtarten:")
print(df["fruit_type"].unique())


# Einzigartige Farben anzeigen
# Prüft vorhandene Kategorien in der Spalte "color"
print("\nFarben:")
print(df["color"].unique())

# Einzigartige Größen anzeigen
# Prüft vorhandene Kategorien in der Spalte "size"
print("\nGrößen:")
print(df["size"].unique())


#-----------------
# Datenbereinigung 
#-----------------

# Objekt der Klasse erstellen
cleaner = DataCleaner(df)

# Unnötige Spalte entfernen
cleaner.remove_columns(["Unnamed: 0"])

# Schreibfehler korrigieren
cleaner.replace_values(
    "size",
    {
        "Largee": "Large"
    }
)

# Ausreißer entfernen
cleaner.remove_outliers(
    "weight",
    0,
    200
)

# Bereinigte Daten zurückgeben
df = cleaner.get_data()

# Bereinigte Daten anzeigen
print(df.head())


#----------------
# FeatureEngineer
#----------------


## Objekt der Klasse erstellen
feature_engineer = FeatureEngineer(df)

# Features erstellen
X = feature_engineer.create_features()

# Zielvariable erstellen
y = feature_engineer.create_target()

# Kontrolle der Daten
print("\nFeatures:")
print(X.head())

print("\nZielvariable:")
print(y.head())


#-------------
# DataSplitter
#-------------

# Objekt der Klasse erstellen
splitter = DataSplitter(X, y)

# Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = splitter.split_data()

# Kontrollausgabe
print("\nAnzahl Trainingsdaten:")
print(len(X_train))

print("\nAnzahl Testdaten:")
print(len(X_test))


#--------------
# PreProcessor
#--------------
# Objekt der Preprocessor-Klasse erstellen
preprocessor_builder = Preprocessor()

# Preprocessor erstellen
preprocessor = preprocessor_builder.create_preprocessor()


#--------------
# ModelTrainer
#--------------

# Objekt der Klasse erstellen
trainer = ModelTrainer(preprocessor)

# Logistische Regression trainieren
logistic_model = trainer.train_logistic_regression(
    X_train,
    y_train
)

# Decision Tree trainieren
tree_model = trainer.train_decision_tree(
    X_train,
    y_train
)


#----------------
# ModelEvaluator
#----------------

# Objekt der Evaluator-Klasse erstellen
evaluator = ModelEvaluator()

# Logistische Regression bewerten
log_accuracy, log_f1 = evaluator.evaluate_model(
    logistic_model,
    X_test,
    y_test,
    "Logistische Regression"
)

# Decision Tree bewerten
tree_accuracy, tree_f1 = evaluator.evaluate_model(
    tree_model,
    X_test,
    y_test,
    "Decision Tree"
)

#-----------------------
# Vergleich der Modelle
#-----------------------
# Das Modell mit den besseren Metriken
# liefert die zuverlässigeren Vorhersagen.

print("\nModellvergleich:")

print("\nLogistische Regression:")
print("Accuracy:", log_accuracy)
print("F1-Score:", log_f1)

print("\nDecision Tree:")
print("Accuracy:", tree_accuracy)
print("F1-Score:", tree_f1)

#-------------------------------
# Bestimmung des besten Modells
#-------------------------------
# Das Modell mit der höheren Accuracy
# wird als besseres Modell ausgewählt.

if tree_accuracy > log_accuracy:

    print("\nDer Decision Tree liefert die bessere Modellleistung.")

else:

    print("\nDie logistische Regression liefert die bessere Modellleistung.")

#------------------
# Modell speichern
#------------------
# Speichert die logistische Regression
# Dadurch kann das Modell später wiederverwendet werden, ohne erneut trainiert werden zu müssen.
joblib.dump(
    logistic_model,
    "logistic_regression_model.joblib"
)

print("\nLogistische Regression wurde gespeichert.")


# Speichert den Decision Tree
joblib.dump(
    tree_model,
    "decision_tree_model.joblib"
)

print("\nDecision Tree wurde gespeichert.")
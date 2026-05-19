import pandas as pd

# Allgemeine Klasse zur Datenbereinigung
class DataCleaner:

    # Konstruktor der Klasse
    # Speichert den Datensatz
    def __init__(self, dataframe):
        self.df = dataframe

    # Diese Methode entfernt angegebene Spalten aus dem Datensatz
    def remove_columns(self, columns):
        
        # Entfernt ausgewählte Spalten
        self.df = self.df.drop(columns=columns)

        print("\nSpalten wurden entfernt.")


    # Diese Methode ersetzt fehlerhafte Werte in einer Spalte
    def replace_values(self, column, replacements):

        # Ersetzt fehlerhafte Kategorien oder Schreibweisen
        self.df[column] = self.df[column].replace(replacements)

        print(f"\nWerte in '{column}' wurden ersetzt.")


    # Diese Methode entfernt numerische Ausreißer
    def remove_outliers(self, column, min_value, max_value):

        # Filtert Werte außerhalb des erlaubten Bereichs
        self.df = self.df[
            (self.df[column] > min_value) &
            (self.df[column] < max_value)
        ]

        print(f"\nAusreißer in '{column}' wurden entfernt.")


    # Diese Methode gibt den bereinigten Datensatz zurück
    def get_data(self):

        # Gibt die bereinigten Daten zurück
        return self.df
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Klasse zur Bewertung der trainierten Modelle
class ModelEvaluator:

    # Diese Methode bewertet ein Modell mit Testdaten
    def evaluate_model(self, model, X_test, y_test, model_name):

        # Erstellt Vorhersagen mit unbekannten Testdaten
        y_pred = model.predict(X_test)

        # Accuracy misst den Anteil korrekt vorhergesagter Klassen
        accuracy = accuracy_score(y_test, y_pred)

        # F1-Score kombiniert Precision und Recall
        # average="macro" bewertet alle Klassen gleich stark
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"\nErgebnisse für {model_name}:")
        print("Accuracy:", accuracy)
        print("F1-Score:", f1)

        # Classification Report zeigt detaillierte Metriken pro Klasse
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return accuracy, f1
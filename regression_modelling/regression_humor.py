import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dramas_df = pd.read_csv('dramas.csv', sep="\t")

X = dramas_df['raw_text']  # Textspalte im DataFrame, das wir aus der csv Datei einlesen
y = dramas_df['Annotation_1']  # Annotationswert-Spalte im DataFrame (Annotation 1: Melina)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)  # Anpassen der Testgröße und des Zufallsstates

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LinearRegression()
model.fit(X_train_vec, y_train)

train_score = model.score(X_train_vec, y_train)
test_score = model.score(X_test_vec, y_test)

print("Trainingsgenauigkeit:", train_score)
print("Testgenauigkeit:", test_score)

path_to_directory = 'postprocessing/dramas'
directory = os.fsencode(path_to_directory)
list_of_unseen_dramas = []
names_of_unseen_dramas = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    if filename.endswith(".txt"):

        path_filename = os.path.join(path_to_directory, filename)
        with open(path_filename, mode='r') as f:
            drama = f.read()
            list_of_unseen_dramas.append(drama)
            names_of_unseen_dramas.append(filename)

new_texts_vec = vectorizer.transform(list_of_unseen_dramas)
predictions = model.predict(new_texts_vec)

results_df = pd.DataFrame({'file_name': names_of_unseen_dramas, 'regression_prediction': predictions})


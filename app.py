from flask import Flask, render_template, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

app = Flask(__name__)

target = 'Race_Wins'
features = ['Race_Entries', 'Race_Starts', 'Pole_Positions', 'Podiums', 'Fastest_Laps', 'Points_Per_Entry', 'Years_Active']

def load_data():
    return pd.read_csv('data/F1DriversDataset.csv')

def preprocess_data(df):
    df.dropna(inplace=True)
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_models(X_train, y_train):
    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=200),
        'SVM': SVC()
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def evaluate_models(models, X_test, y_test):
    evaluation = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        evaluation[name] = accuracy
    return evaluation

def visualize_results(evaluation):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(evaluation.keys()), y=list(evaluation.values()))
    plt.title('Model Accuracies')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.savefig('static/model_accuracies.png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    models = train_models(X_train, y_train)
    evaluation = evaluate_models(models, X_test, y_test)
    visualize_results(evaluation)
    return redirect(url_for('results'))

@app.route('/results', methods=['GET'])
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)

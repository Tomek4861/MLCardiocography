import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, \
    classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from utils import RANDOM_SEED
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


class ModelEvaluator:
    def __init__(self):

        self.dataframe = pd.read_csv('cardiotocography_filled.csv', encoding='utf-8')
        self.train_df, self.test_df = train_test_split(self.dataframe, test_size=0.2, random_state=RANDOM_SEED)

        target = "CLASS"
        self.x_train = self.train_df.drop(columns=[target,])
        self.y_train = self.train_df[target]

        self.x_test = self.test_df.drop(columns=[target,])
        self.y_test = self.test_df[target]

        self.y_pred = None


    def evaluate(self, model):
        model.fit(self.x_train, self.y_train)
        self.y_pred = model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, self.y_pred)
        print("Confusion matrix:\n", cm)

        acc = accuracy_score(self.y_test, self.y_pred)
        print(f"Accuracy: {acc:.4f}")


        precision = precision_score(self.y_test, self.y_pred, average='macro')
        recall = recall_score(self.y_test, self.y_pred, average='macro')
        f1 = f1_score(self.y_test, self.y_pred, average='macro')
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall    (macro): {recall:.4f}")
        print(f"F1-score  (macro): {f1:.4f}")

        print("\nFull classification report:\n",
              classification_report(self.y_test, self.y_pred))

    def evaluate_grid_search(self, grid_search_obj: GridSearchCV,):
        grid_search_obj.fit(self.x_train, self.y_train)
        print("Best params:", grid_search_obj.best_params_)
        print("Best CV F1:", grid_search_obj.best_score_)

        best_dt = grid_search_obj.best_estimator_
        self.y_pred = best_dt.predict(self.x_test)

        print("Test accuracy:", accuracy_score(self.y_test, self.y_pred))
        print("\nClassification report:\n", classification_report(self.y_test, self.y_pred))
        print("\nConfusion matrix:\n", confusion_matrix(self.y_test, self.y_pred))



def run_models():

    me = ModelEvaluator()
    print('-' * 25 + "No Scaling" + '-' * 25)
    param_grid_dt = {
        'max_depth':        [None, 5, 8, 12, 15],
        'min_samples_leaf': [1, 5, 10, 13],
        'min_samples_split': [2, 5, 10],

    }
    gs_dt = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_SEED),
        param_grid_dt, cv=5, scoring='f1_macro', n_jobs=2, verbose=1
    )
    print(5 * "=" + " Decision Tree " + 5 * "=")
    me.evaluate_grid_search(gs_dt)

    param_grid_nb = {
        'var_smoothing': [1e-15, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3],
    }
    gs_nb = GridSearchCV(
        GaussianNB(),
        param_grid_nb, cv=5, scoring='f1_macro', n_jobs=2, verbose=1
    )
    print(5 * "=" + "Gaussian Naive Bayes " + 5 * "=")
    me.evaluate_grid_search(gs_nb)

    print('-' * 25 + "Scaling MinMax" + '-' * 25)
    pipe_decision_tree = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])

    param_grid_dt = {
        'clf__max_depth':        [None, 5, 8, 12, 15],
        'clf__min_samples_leaf': [1, 5, 10, 13],
        'clf__min_samples_split': [2, 5, 10],

    }
    gs_dt = GridSearchCV(
        estimator=pipe_decision_tree,
        param_grid=param_grid_dt, cv=5, scoring='f1_macro', n_jobs=2, verbose=1
    )
    print(5 * "=" + "Decision Tree Scaling" + 5 * "=")
    me.evaluate_grid_search(gs_dt)

    pipe_bayes = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', GaussianNB())
    ])

    param_grid_nb = {
        'clf__var_smoothing': [1e-15, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3],
    }
    gs_nb = GridSearchCV(
        estimator=pipe_bayes,
        param_grid=param_grid_nb, cv=5, scoring='f1_macro', n_jobs=2, verbose=1
    )
    print(5 * "=" + "Gaussian Naive Bayes Scaling" + 5 * "=")
    me.evaluate_grid_search(gs_nb)



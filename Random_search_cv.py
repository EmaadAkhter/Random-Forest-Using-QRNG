import time
from random_forest import random_forest
from quantum_random_number_genrator import qml_random_choice, NumberGenerator
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import math
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, make_classification


class Random_search_cv:
    def __init__(self, random_forest_class, param_distributions, cv=5, n_iter=50, scoring='accuracy',
                 random_state=None):
        self.random_forest_class = random_forest_class
        self.param_distributions = param_distributions
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.random_state = random_state

        self.best_params_ = None
        self.best_score_ = -np.inf
        self.cv_results_ = []

    def _sample_params(self):
        params = {}
        for param_name, param_range in self.param_distributions.items():
            if isinstance(param_range, list):
                n_options = len(param_range)
                generator = NumberGenerator(2 ** max(1, int(np.ceil(np.log2(n_options)))))

                while True:
                    rand_val = generator.generate_batch_unbiased(1)[0]
                    if rand_val < n_options:
                        params[param_name] = param_range[rand_val]
                        break

            elif isinstance(param_range, tuple) and len(param_range) == 2:
                min_val, max_val = param_range

                n_bit = 10
                maxint = 2 ** n_bit - 1
                generator = NumberGenerator(maxint + 1)
                randint = generator.generate_batch_unbiased(1)[0]
                random_float = randint / maxint

                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = int(min_val + random_float * (max_val - min_val))
                else:
                    params[param_name] = min_val + random_float * (max_val - min_val)

        return params

    def _cross_val_score(self, params, X, y):
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scores = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            rf = self.random_forest_class(**params)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)

            if self.scoring == 'accuracy':
                score = accuracy_score(y_test, y_pred)
            elif self.scoring == 'f1':
                score = f1_score(y_test, y_pred, average='weighted')
            else:
                raise ValueError("scoring must be accuracy or f1")

            scores.append(score)

        return scores

    def fit(self, X, y):
        for i in range(self.n_iter):
            params = self._sample_params()
            scores = self._cross_val_score(params, X, y)
            mean = np.mean(scores)
            std = np.std(scores)

            result = {
                'params': params.copy(),
                'mean_test_score': mean,
                'std_test_score': std,
                'rank': None
            }

            self.cv_results_.append(result)

            if mean > self.best_score_:
                self.best_score_ = mean
                self.best_params_ = params.copy()

            print(f"Iteration {i + 1}/{self.n_iter}: Score = {mean:.4f} (+/- {std:.4f})")
            print(f"  Parameters: {params}")

        self.cv_results_.sort(key=lambda x: x['mean_test_score'], reverse=True)
        for rank, result in enumerate(self.cv_results_):
            result['rank'] = rank + 1
        return self

    def get_best_estimator(self):
        if self.best_params_ is None:
            raise ValueError("Must call fit() before getting best estimator")

        return self.random_forest_class(**self.best_params_)

    def get_quantum_feature_selection(self, X, n_features):
        n_total_features = X.shape[1]
        selected_idx = qml_random_choice(n_total_features, n_features, replace=False)
        return np.array(selected_idx)


def create_example_param_distributions():
    return {
        'n_trees': [i for i in range(1, 200, 5)],
        'max_depth': [i for i in range(1, 200, 5)],
        'min_sample_split': [i for i in range(1, 200, 5)],
        'n_features': [i for i in range(0, 200, 5)]
    }


if __name__ == "__main__":
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    print(X.shape)
    print(y.shape)
    start_time = time.time()

    param_distributions = create_example_param_distributions()

    quantum_search = Random_search_cv(random_forest, param_distributions, cv=5, n_iter=100, scoring='accuracy',
                                      random_state=42)

    quantum_search.fit(X, y)

    best_rf = quantum_search.get_best_estimator()

    selected_features = quantum_search.get_quantum_feature_selection(X, n_features=10)
    X_selected = X[:, selected_features]


    print(f"Best parameters: {quantum_search.best_params_}")
    print(f"Best score: {quantum_search.best_score_:.4f}")
    end_time=time.time()
    print(f"Time: {end_time-start_time}s")
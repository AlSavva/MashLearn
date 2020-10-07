from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, \
    cross_val_score, KFold
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
def train_grid_search(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
    #kfold = KFold(n_splits=5, random_state=10, shuffle = True)
    grid_searcher = GridSearchCV(
    KNeighborsClassifier(),
    param_grid = {
        'n_neighbors': list(range(1,21)),
    }
)
    grid_searcher.fit(X_train, y_train)
    grid_searcher.predict(X_test)
    return grid_searcher.cv_results_['mean_test_score']    
    

mean_test_scores = []
for i in range(100):
  X, y = make_moons(n_samples=1000, noise=0.5)
  mean_test_score = train_grid_search(X, y)
  mean_test_scores.append(mean_test_score)

mean_test_scores = np.array(mean_test_scores)
plt.plot(np.arange(1, 21), np.mean(mean_test_scores, axis=0))

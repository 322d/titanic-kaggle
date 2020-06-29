import features
import pandas as pd

from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, features.train_data, target,
                        cv=k_fold, n_jobs=1, scoring=scoring)

print(round(np.mean(score)*100, 2))

clf.fit(features.train_data, features.target)
test_data = features.test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": prediction
})

submission.to_csv('submission.csv', index=False)

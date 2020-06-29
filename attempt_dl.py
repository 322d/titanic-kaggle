import features

import pandas as pd
import numpy as np

from keras import layers
from keras import models

X_train = features.X_train.to_numpy()
y_train = features.y_train.to_numpy()
X_train.flatten()
y_train.flatten()

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(8,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=8, batch_size=1, verbose=1)

y_test = pd.read_csv('submission.csv')
y_test = y_test.drop('PassengerId', axis=1)
y_test = y_test.to_numpy()
y_test.flatten()

features.test = features.test.drop('PassengerId', axis=1)
X_test = features.test.to_numpy()
X_test.flatten()

y_pred = model.predict(X_test)
submission = pd.DataFrame(y_pred)
submission.insert(0, 'PassengerId', range(892, 892+418))
submission = submission.rename(columns={0: "Survived"})
submission['Survived'] = round(submission['Survived'])
submission['Survived'] = submission['Survived'].astype(np.int64)

score = model.evaluate(X_test, y_test, verbose=1)
print(score)

submission.to_csv('submission_dl.csv', index=False)

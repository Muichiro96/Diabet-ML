import numpy as np


class Predictor:
    def __init__(self):
        # Importing the libraries
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
        from sklearn.neighbors import KNeighborsClassifier
        # Importing the dataset
        dataset = pd.read_csv('diabetes.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 8].values

        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                            random_state = 42)

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler()
        X_train = self.sc.fit_transform(X_train)
        X_test = self.sc.transform(X_test)


        # Parameter evaluation
        knnclf = KNeighborsClassifier()
        parameters={'n_neighbors': range(1, 20)}
        gridsearch=GridSearchCV(knnclf, parameters, cv=100, scoring='roc_auc')
        gridsearch.fit(X, y)
        print(gridsearch.best_params_)
        print(gridsearch.best_score_)

        # Fitting K-NN to the Training set
        knnClassifier = KNeighborsClassifier(n_neighbors = 18)
        knnClassifier.fit(X_train, y_train)
        print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knnClassifier.score(X_train, y_train)))
        print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knnClassifier.score(X_test, y_test)))

        # Predicting the Test set results
        y_pred = knnClassifier.predict(X_test)

        # Making the Confusion Matrix
        from sklearn.metrics import classification_report, confusion_matrix
        cm = confusion_matrix(y_test, y_pred)

        print('TP - True Negative {}'.format(cm[0,0]))
        print('FP - False Positive {}'.format(cm[0,1]))
        print('FN - False Negative {}'.format(cm[1,0]))
        print('TP - True Positive {}'.format(cm[1,1]))
        print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))))
        print('Misclassification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))))

        round(roc_auc_score(y_test,y_pred),5)

        import numpy as np

        # Example input data (replace this with your actual input data)
        new_data = np.array([7,81,78,40,48,46.7,0.261,42])

        # Reshape the input data to make it 2D
        new_data_reshaped = new_data.reshape(1, -1)

        # Scale the input data using the same StandardScaler
        scaled_new_data = self.sc.transform(new_data_reshaped)

        # Make predictions
        new_data_probabilities = knnClassifier.predict(scaled_new_data)

        print('Probability scores for each class:', new_data_probabilities)
        self.predictor = knnClassifier
    def isDiabetic(self, inputData):
        # Example input data (replace this with your actual input data)
        # stdin = input("Give input")
        # splittedArray = stdin.split(",")
        #splittedArray = [float(x) for x in splittedArray]
        # inputData = splittedArray
        # print(inputData)
        new_data = np.array(inputData)

        # Reshape the input data to make it 2D

        new_data_reshaped = new_data.reshape(1, -1)

        # Scale the input data using the same StandardScaler
        scaled_new_data = self.sc.transform(new_data_reshaped)
        result = self.predictor.predict(scaled_new_data)
        print(result)
        return bool(result[0])


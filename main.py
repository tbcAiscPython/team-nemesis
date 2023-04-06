import pandas as pd             #imports the pandas library and renames it as "pd".
import re                       #regular expression library that provides powerful tools for pattern matching and string manipulation.

import nltk                    #nltk library provides tools and resources for working with human language data.
#nltk.download()                #This function opens a GUI (graphical user interface) that allows the user to select which data and resources they want to download from the nltk library.
from nltk.corpus import stopwords  # imports the stopwords corpus from the nltk library. The stopwords corpus is a collection of common stopwords for different languages that can be used to remove these words from text data.
from nltk.stem import WordNetLemmatizer  # Lemmatization is the process of reducing a word to its base or root form, which can be useful for reducing the number of unique words in a text corpus.

from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer is a method for converting text data into a matrix of token counts, which is a common way of representing text data in machine learning applications.
from sklearn.model_selection import GridSearchCV            #GridSearchCV is a method for tuning hyperparameters of a machine learning model using a grid search over a specified parameter space.
from sklearn.ensemble import RandomForestClassifier         #imports the RandomForestClassifier class from the sklearn.ensemble module. RandomForestClassifier is a type of ensemble learning algorithm that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control overfitting.
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,classification_report
import ssl
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import itertools
# train a random forest classifier
from sklearn.ensemble import RandomForestClassifier


class Sentiment_Analysis:

    def __init__(self):
        # set the SSL certificate to default to avoid errors
        ssl._create_default_https_context = ssl._create_unverified_context

        # specify the packages to download
        packages = ['stopwords', 'wordnet']

        # download the packages
        for package in packages:
            nltk.download(package)

    def get_data(self):
        __pipeline__ = []
        missing_values_rows = df.isnull().sum(
            axis=1)  # using the isnull() method to create a boolean mask that indicates where the missing values are, and then use the sum() method to count the number of missing values in each row.
        df = df.dropna()  # dropping the rows which has null values in rows
        # converting numbers into sentiment using replace() func.
        df['output'].replace([1, 2, 3, 4, 5], ['negative', 'negative', 'neutral', 'positive', 'positive'], inplace=True)
        # This can make the data more interpretable and easier to work with in subsequent analysis.
        # Encoding/ or replacing again as the machine can take numbers and not words.
        # Encoding with respect to our needs and for easy visualisation.
        df['output'].replace(['positive', 'neutral', 'negative'], [1, 2, 3], inplace=True)
        return df

    def text_transformation(self,df_col):
        lm = WordNetLemmatizer()
        corpus = []  # A new empty list, corpus, is created to store the transformed text data.
        for review in df_col:
            new_review = re.sub('[^a-zA-Z]', ' ', str(review))
            new_review = new_review.lower()  # transformed text data is then converted to lowercase
            new_review = new_review.split()  # transformed text data is then split into individual words
            new_review = [lm.lemmatize(word) for word in new_review if word not in set(stopwords.words('english'))]
            corpus.append(' '.join(str(x) for x in
                                   new_review))  # The lemmatized and filtered words are then joined back into a single string using the join() method, with a space as the separator. The resulting string is added to the corpus list using the append() method.
        return corpus  # returns the list of transformed text data, corpus.

    def preprocess(self):
        df = self.get_data()
        corpus = self.text_transformation(df['reviews_title'])
        cv = CountVectorizer(ngram_range=(1, 2))
        train_data = cv.fit_transform(corpus)
        return cv, train_data

    def train(self):
        try:
            df = self.get_data()
            df['output'].replace([1, 2, 3, 4, 5], ['negative', 'negative', 'neutral', 'positive', 'positive'],
                                 inplace=True)
            # Encoding----
            df['output'].replace(['positive', 'neutral', 'negative'], [1, 2, 3], inplace=True)
            train_data = self.preprocess()[1]
            X = train_data
            y = df.output
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
            rf = RandomForestClassifier(max_features='sqrt',
                                        max_depth=None,
                                        n_estimators=500,
                                        min_samples_split=5,
                                        min_samples_leaf=2,
                                        bootstrap=False)
            rf.fit(X, y)
            y_pred = rf.predict(X_test)
            score = accuracy_score(y_test, y_pred)

        except Exception as e:
            print(f'Error creating model ')

    def sentiment_predictor(self, review):
        """

        Args:
            review: 'sentence/review'

        Returns:Sentiment and  keywords from reviews and the specific sentiment

        """

        def expression_check(prediction_input):
            if prediction_input == 1:
                return "Positive"
            elif prediction_input == 2:
                return "Neutral"
            else:
                return "Negative"

        input = text_transformation(input)  # input text is passed to the function named as text_transformation
        transformed_input = cv.transform(input)  # transforming input text into a matrix of token counts
        prediction = rf.predict(transformed_input)  # storing the predicted values in 'prediction'
        expression_check(prediction)  # calling the function to print the final sentiment.
        return prediction

df_pred = 'Get data from AWS'
df_pred['predicted_sentiment'] = sentiment_predictor(df_pred["user_review"])
df_pred["prediction"] = ""
for index, row in df_pred.iterrows():
# Apply the "sentiment_predictor" function to the "user_review" column of that row
    prediction = sentiment_predictor(row["user_review"])
    # Store the predicted sentiment value in the "prediction" column of that row
    df_pred.loc[index, "prediction"] = prediction


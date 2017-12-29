import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def main():
    data_dir_path = './data'

    # Import `fake_or_real_news.csv`
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    # Inspect shape of `df`
    print(df.shape)

    # Print first lines of `df`
    print(df.head())

    # Set index
    df = df.set_index("Unnamed: 0")

    # Print first lines of `df`
    print(df.head())

    # Set `y`
    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    count_test = count_vectorizer.transform(X_test)

    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Get the feature names of `tfidf_vectorizer`
    print(tfidf_vectorizer.get_feature_names()[-10:])

    # Get the feature names of `count_vectorizer`
    print(count_vectorizer.get_feature_names()[:10])

    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

    print(count_df.head())

    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

    print(tfidf_df.head())

    print(count_df.equals(tfidf_df))

    difference = set(count_df.columns) - set(tfidf_df.columns)
    print(difference)


if __name__ == '__main__':
    main()

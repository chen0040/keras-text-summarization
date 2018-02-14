import pandas as pd
from sklearn.model_selection import train_test_split
from keras_text_summarization.library.applications.fake_news_loader import fit_text


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
    Y = df.title
    X = df['text']

    # Drop the `label` column
    df.drop("title", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=53)

    print('X train: ', X_train.shape)
    print('Y train: ', y_train.shape)

    config = fit_text(X, Y)

    print('num_input_tokens: ', config['num_input_tokens'])
    print('num_target_tokens: ', config['num_target_tokens'])
    print('max_input_seq_length: ', config['max_input_seq_length'])
    print('max_target_seq_length: ', config['max_target_seq_length'])


if __name__ == '__main__':
    main()

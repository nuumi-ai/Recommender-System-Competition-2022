import numpy as np
import pandas as pd


def evaluate(predictions, ratings, weights):
    """
    calculated weighted MSE with numpy, the smaller the better
    :param predictions: nx1 matrix
    :param ratings: nx1 matrix
    :param weights: nx1 matrix
    :return:
    """
    error = np.dot(((predictions - ratings) ** 2), weights.reshape((1, -1))).mean()

    return error


def get_weights(df):
    rating_counts = np.array([len(df[df['rating'] == i]) for i in [1, 2, 3, 4, 5]])
    inverse_count = 1 / rating_counts
    normalized_inverse_count = inverse_count / np.linalg.norm(inverse_count)

    return normalized_inverse_count


def main(p_file, rating_file):
    predictions = open(p_file, 'r').readlines()
    predictions = np.array([float(p) for p in predictions]).reshape((-1, 1))

    target_df = pd.read_csv(rating_file, sep='\t')['rating']
    targets = np.array(target_df).reshape((-1, 1))
    weights = get_weights(target_df).reshape((-1, 1))

    print(f"Error = {evaluate(predictions, targets, weights)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='pred')
    parser.add_argument('-t', dest='target')
    args = parser.parse_args()

    main(args.pred, args.target)

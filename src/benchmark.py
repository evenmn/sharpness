import argparse

from metrics import compute_metric, compute_all_metrics

parser = argparse.ArgumentParser(description='Sharpness Benchmarks')

parser.add_argument('--fakedata', action='store_true',
                    help='use fake data')

parser.add_argument('--data-filename', type=str, default='../data/something.npy',
                    help='name of file to load data from')
parser.add_argument('--output-filename', type=str, default='../media/output.jpg',
                    help='name of output file visualization to save data')

parser.add_argument('--metric', type=str, default='all',
                    help='evaluation metric to compute')


def visualize(data, output_filename=None):
    pass


def generate_synthetic_data():
    pass


def load_real_data():
    pass


def load(args):
    if args.fakedata:
        X = generate_synthetic_data()
    else:
        X = load_real_data(args.data_filename)
    return X


def transform(X):
    T = None
    return T


def evaluate(X, T, metric):
    if metric == 'all':
        results = compute_all_metrics(X, T)
        for metric, result in results.items():
            print(f'{metric}: {result}')
    else:
        metric_value = compute_metric(X, T, metric)
        print(f'{metric}: {metric_value}')


def main(args):
    X = load(args)
    T = transform(X)
    metrics = evaluate(X, T)

    data = dict()
    data['X'] = X
    data['T'] = T
    data['metrics'] = metrics

    visualize(data, args.output_filename)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

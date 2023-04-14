import argparse

parser = argparse.ArgumentParser(description='Sharpness Benchmarks')

parser.add_argument('--fakedata', action='store_true',
                    help='use fake data')

parser.add_argument('--data-filename', type=str, default='../data/something.npy',
                    help='name of file to load data from')
parser.add_argument('--output-filename', type=str, default='../media/output.jpg',
                    help='name of output file visualization to save data')


def visualize(data, output_filename=None):
    pass


def generate_synthetic_data():
    pass


def load_real_data():
    pass


def load(args):
    if args.fakedata:
        x = generate_synthetic_data()
    else:
        x = load_real_data(args.data_filename)
    return x


def transform(x):
    t = None
    return t


def eval(x, t):
    pass


def main(args):
    x = load(args)
    t = transform(x)
    metrics = eval(x, t)

    data = dict()
    visualize(data, args.output_filename)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

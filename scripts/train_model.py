import argparse
from torch import nn, optim, device, cuda, save, load, manual_seed
import random
import numpy as np
import sys
import os
# get the current_file_dir
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# get the parent_dir
parent_dir = os.path.abspath(os.path.join(current_file_dir, os.pardir))
# Add the parent directory to the path list
sys.path.append(parent_dir)

from detect.util import get_data, get_model


device = device("cuda" if cuda.is_available() else "cpu")


# 保持实验的一致性
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)


# 随机种子取21
seed_everything(21)


def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset parameter must be either 'mnist', 'cifar', or 'svhn'"
    print('Data set: %s' % args.dataset)
    train_dataset, test_dataset = get_data(args.dataset, args.batch_size)
    model = get_model(args.dataset)
    model = model.to(device)
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(args.learning_rate, args.weight_decay)
    for i in range(args.epochs):
        all_counts = 0
        correct_counts = 0
        for images, labels in train_dataset:
            images = images.to(device)
            labels = labels.to(device)
            results = model(images)
            loss = loss_function(results, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_counts += len(labels)
            correct_counts += (results.argmax(axis=1) == labels).sum()
        print(f'epochs:{i}  train_accuracy:{round(float(correct_counts/all_counts*100), 2)}')
    save(model, f'../data/model/model_{args.dataset}.pt')
    # model = load(f'../data/model_{args.dataset}.pt')
    model.eval()
    all_counts = 0
    correct_counts = 0
    for images, labels in test_dataset:
        images = images.to(device)
        labels = labels.to(device)
        results = model(images)
        all_counts += len(labels)
        correct_counts += (results.argmax(axis=1) == labels).sum()
    print(f'test_accuracy:{round(float(correct_counts / all_counts * 100), 2)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True,
        type=str,
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False,
        type=int,
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False,
        type=int,
    )
    parser.add_argument(
        '-l', '--learning_rate',
        help="The learning rate to use for training.",
        required=False,
        type=int,
    )
    parser.add_argument(
        '-w', '--weight_decay',
        help="To mitigate the over fitting problem",
        required=False,
        type=float
    )
    parser.set_defaults(epochs=20)
    parser.set_defaults(batch_size=128)
    parser.set_defaults(learning_rate=0.001)
    parser.set_defaults(weight_decay=1e-5)
    args = parser.parse_args()
    main(args)

# mnist: epochs=20, batch_size=128, learning_rate=0.001, weight_decay=1e-5
# cifar10 epochs=100, batch_size=128, learning_rate=0.001, weight_decay=1e-5

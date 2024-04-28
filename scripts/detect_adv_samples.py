import argparse
import os
from torch import load, cat, tensor, device, cuda, float32, norm, where
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.neighbors import KernelDensity


current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_dir, os.pardir))
sys.path.append(parent_dir)


from detect.util import get_data, get_noisy_samples, get_mc_predictions, get_deep_representations, score_samples, \
    normalize, train_lr, compute_roc


# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00}

device = device("cuda" if cuda.is_available() else "cpu")


def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim', 'pgd', 'jsma', 'cw', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim', 'pgd' " \
        "'jsma' or 'cw'"
    assert os.path.isfile('../data/model/model_%s.pt' % args.dataset), \
        'model file not found... must first train model using train_model.py.'
    assert os.path.isfile('../data/adv_samples/Adv_%s_%s.pt' %
                          (args.dataset, args.attack)), \
        'adversarial sample file not found... must first craft adversarial ' \
        'samples using craft_adv_samples.py'
    print('Loading the data and model...')
    # Load the model
    model = load(f'../data/model/model_{args.dataset}.pt')
    model.eval()
    # Load the dataset
    train_dataset, test_dataset = get_data(args.dataset, args.batch_size)
    # Check attack type, select adversarial and noisy samples accordingly
    print('Loading noisy and adversarial samples...')
    if args.attack == 'all':
        # TODO: implement 'all' option
        # test_adv = ...
        # test_noisy = ...
        raise NotImplementedError("'All' types detector not yet implemented.")
    else:
        # Splicing test_dataset
        x_test = []
        y_test_pre = []
        for images, labels in test_dataset:
            x_test.append(images)
        x_test = cat(x_test, dim=0)
        y_test = load('../data/adv_samples/Adv_%s_%s.pt' % (args.dataset, args.attack))['labels']
        # Load adversarial samples
        x_test_adv = load('../data/adv_samples/Adv_%s_%s.pt' % (args.dataset, args.attack))['adv_inputs']
        y_test_adv_pre = []
        # Craft an equal number of noisy samples
        x_test_noisy = get_noisy_samples(x_test, x_test_adv, args.dataset, args.attack)
        y_test_noisy_pre = []

    # Check model accuracies on each sample type
    for s_type, dataset in zip(['normal', 'noisy', 'adversarial'],
                               [x_test, x_test_noisy, x_test_adv]):
        all_count = 0
        correct_count = 0
        if s_type == 'normal':
            for images, labels in test_dataset:
                images = images.to(device)
                labels = labels.to(device)
                results = model(images)
                y_test_pre.append(results)
                all_count += len(labels)
                correct_count += float((results.argmax(axis=1) == labels).sum())
            acc = round(correct_count/all_count*100, 2)
        else:
            for i in range(len(dataset)):
                image = tensor(dataset[i].to(device).unsqueeze(0), dtype=float32)
                label = y_test[i].to(device)
                result = model(image)
                if s_type == 'noisy':
                    y_test_noisy_pre.append(result)
                else:
                    y_test_adv_pre.append(result)
                all_count += 1
                correct_count += float((result.argmax(axis=1) == label).sum())
            acc = round(correct_count / all_count*100, 2)

        print("Model accuracy on the %s test set: %0.2f%%" %
              (s_type, acc))
        # # Compute and display average perturbation sizes
        if not s_type == 'normal':
            l2_diff = norm(
                dataset.reshape((len(x_test), -1)) -
                x_test.reshape((len(x_test), -1)),
                p=2,
                dim=1
            ).mean()
            print("Average L-2 perturbation size of the %s test set: %0.2f" %
                  (s_type, l2_diff))

    # Refine the normal, noisy and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    y_test_pre = cat(y_test_pre, dim=0)
    y_test_noisy_pre = cat(y_test_noisy_pre, dim=0)
    y_test_adv_pre = cat(y_test_adv_pre, dim=0)
    inds_correct = where(y_test.to(device) == y_test_pre.argmax(axis=1))[0].to('cpu')
    x_test = x_test[inds_correct]
    x_test_noisy = x_test_noisy[inds_correct]
    x_test_noisy = x_test_noisy.float()
    x_test_adv = x_test_adv[inds_correct]
    y_test = y_test[inds_correct]
    y_test_pre = y_test_pre[inds_correct]
    y_test_noisy_pre = y_test_noisy_pre[inds_correct]
    y_test_adv_pre = y_test_adv_pre[inds_correct]

    # Get Bayesian uncertainty scores
    # The uncertainty is nothing like the original experiment, I don't know what was done wrong.
    print('Getting Monte Carlo dropout variance predictions...')
    uncerts_normal = get_mc_predictions(model, x_test, y_test,
                                        batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)
    uncerts_noisy = get_mc_predictions(model, x_test_noisy, y_test,
                                       batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)
    uncerts_adv = get_mc_predictions(model, x_test_adv, y_test,
                                     batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)
    print(uncerts_normal.mean())
    print(uncerts_noisy.mean())
    print(uncerts_adv.mean())
    print((uncerts_normal < uncerts_adv).sum() / uncerts_normal.size)

    # Get KDE scores
    # Get deep feature representations
    print('Getting deep feature representations...')
    x_train = []
    y_train = []
    for images, labels in train_dataset:
        x_train.append(images)
        y_train.append(labels)
    x_train = cat(x_train, dim=0)
    y_train = cat(y_train, dim=0)
    y_train = y_train.numpy()
    # Attention!This will modify the architecture of the original model, as the passed parameter here is a shallow copy,
    # but given that the model is no longer used later, it is straightforward to modify it this way.
    model.net.pop(-1)
    x_train_features = get_deep_representations(model, x_train,
                                                batch_size=args.batch_size)
    x_test_normal_features = get_deep_representations(model, x_test,
                                                      batch_size=args.batch_size)
    x_test_noisy_features = get_deep_representations(model, x_test_noisy,
                                                     batch_size=args.batch_size)
    x_test_adv_features = get_deep_representations(model, x_test_adv,
                                                   batch_size=args.batch_size)

    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    # Since both the MNIST and CIFAR10 datasets have only ten classes, here it is straightforward to use range(10)
    for i in range(10):
        class_inds[i] = np.where(y_train == i)[0]
    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined "
                  "optimal for the specific CNN models of the paper. If you've "
                  "changed your model, you'll need to re-optimize the "
                  "bandwidth.")
    for i in range(10):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=BANDWIDTHS[args.dataset]) \
            .fit(x_train_features[class_inds[i]])

    # Get model predictions
    print('Computing model predictions...')
    preds_test_normal = np.asarray(y_test_pre.argmax(axis=1).detach().cpu())
    preds_test_noisy = np.asarray(y_test_noisy_pre.argmax(axis=1).detach().cpu())
    preds_test_adv = np.asarray(y_test_adv_pre.argmax(axis=1).detach().cpu())

    # Get density estimates
    # Since the model architecture was changed, re-selecting the bandwidth was necessary,
    # but here the optimal bandwidth was not re-searched, which may have been less effective
    print('Computing densities...')
    densities_normal = score_samples(
        kdes,
        x_test_normal_features,
        preds_test_normal
    )
    densities_noisy = score_samples(
        kdes,
        x_test_noisy_features,
        preds_test_noisy
    )
    densities_adv = score_samples(
        kdes,
        x_test_adv_features,
        preds_test_adv
    )
    print((densities_normal > densities_adv).sum())
    print((densities_normal > densities_noisy).sum())
    print((densities_noisy > densities_adv).sum())
    print((densities_normal > densities_adv).sum() / densities_normal.size)

    # Z-score the uncertainty and density values
    uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z = normalize(
        uncerts_normal,
        uncerts_adv,
        uncerts_noisy
    )
    densities_normal_z, densities_adv_z, densities_noisy_z = normalize(
        densities_normal,
        densities_adv,
        densities_noisy
    )

    # Build detector
    values, labels, lr = train_lr(
        densities_pos=densities_adv_z,
        densities_neg=np.concatenate((densities_normal_z, densities_noisy_z)),
        uncerts_pos=uncerts_adv_z,
        uncerts_neg=np.concatenate((uncerts_normal_z, uncerts_noisy_z))
    )

    # Evaluate detector
    # Compute logistic regression model predictions
    probs = lr.predict_proba(values)[:, 1]
    # Compute AUC
    n_samples = len(x_test)
    # The first 2/3 of 'probs' is the negative class (normal and noisy samples),
    # and the last 1/3 is the positive class (adversarial samples).
    _, _, auc_score = compute_roc(
        probs_neg=probs[:2 * n_samples],
        probs_pos=probs[2 * n_samples:]
    )
    print('Detector ROC-AUC score: %0.4f' % auc_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim', 'pgd', 'jsma' 'cw' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=256)
    args = parser.parse_args()
    main(args)

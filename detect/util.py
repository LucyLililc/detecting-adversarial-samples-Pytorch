from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, randn, tensor, zeros_like, zeros, cat, device, cuda
import numpy as np
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp


# Gaussian noise scale sizes that were determined so that the average
# L-2 perturbation size is equal to that of the adversarial samples
# These figures are based on experiments.
STDEVS = {
    'mnist': {'fgsm': 0.310, 'bim': 0.195, 'pgd': 0.235, 'cw': 0.075},
    'cifar': {'fgsm': 0.050, 'bim': 0.022, 'pgd': 0.031, 'cw': 0.005},
    'svhn': {'fgsm': 0.132, 'bim': 0.122}
}
# Set random seed
np.random.seed(0)
# Set device
device = device("cuda" if cuda.is_available() else "cpu")


def get_data(dataset='mnist', batch_size=128):
    """
    TODO
    :param dataset:
    :param batch_size:
    :return:
    """
    assert dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    if dataset == 'mnist':
        # train_dataset includes train_data and train_label
        train_dataset = datasets.MNIST(root='../data/MNIST_Dataset', train=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Resize([28, 28]),
                                       ]), download=True,)
        train_dataset = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=False,)

        test_dataset = datasets.MNIST(root='../data/MNIST_Dataset', train=False,
                                      transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Resize([28, 28]),
                                       ]), download=True,)
        test_dataset = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,)
    elif dataset == 'cifar':
        train_dataset = datasets.CIFAR10(root='../data/CIFAR10_Dataset', train=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ]), download=True,)
        train_dataset = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=False,)

        test_dataset = datasets.CIFAR10(root='../data/CIFAR10_Dataset', train=False,
                                        transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ]), download=True,)
        test_dataset = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, )
    # TODO svhn needs to find

    return train_dataset, test_dataset


# Defining the model architecture on the MNIST dataset
class Model_MNIST(nn.Module):
    def __init__(self):
        super(Model_MNIST, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0,),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,),
            nn.Dropout(p=0.5,),
            nn.Flatten(),
            nn.Linear(in_features=9216, out_features=128, bias=True,),
            nn.ReLU(),
            nn.Dropout(p=0.5,),
            nn.Linear(128, 10,),
            # not use softmax, because the cross entropy loss of pytorch will use softmax
        )

    def forward(self, inputs):
        outputs = self.net(inputs)
        return outputs


# Defining the model architecture on the CIFAR10 dataset
class Model_Cifar10(nn.Module):
    def __init__(self):
        super(Model_Cifar10, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1,),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,),
            nn.Flatten(),
            nn.Dropout(p=0.5,),
            nn.Linear(in_features=2048, out_features=1024,),
            nn.ReLU(),
            nn.Dropout(p=0.5,),
            nn.Linear(in_features=1024, out_features=512,),
            nn.ReLU(),
            nn.Dropout(p=0.5,),
            nn.Linear(in_features=512, out_features=256,),
            nn.ReLU(),
            nn.Dropout(p=0.5,),
            nn.Linear(in_features=256, out_features=128,),
            nn.ReLU(),
            nn.Dropout(p=0.5, ),
            nn.Linear(in_features=128, out_features=10,),
        )

    def forward(self, inputs):
        outputs = self.net(inputs)
        return outputs


def get_model(dataset='mnist'):
    if dataset == 'mnist':
        # MNIST model
        model = Model_MNIST()
        return model
    elif dataset == 'cifar':
        # CIFAR10 Model
        model = Model_Cifar10()
        return model


def flip(x, nb_diff):
    """
    Helper function for get_noisy_samples
    :param x:
    :param nb_diff:
    :return:
    """
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < 0.99)[0]
    assert candidate_inds.shape[0] >= nb_diff
    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = 1.

    return np.reshape(x, original_shape)


def get_noisy_samples(x_test, x_test_adv, dataset, attack):
    # Since the cw attack using the torchattacks library will change almost all pixel values (to a slight degree),
    # the flip operation will not be possible here.
    # Therefore, here, for the time being, the jsma,
    # cw construction of noise samples is replaced by the addition of Gaussian noise
    # (based on the corresponding L2 loss values)
    # if attack in ['jsma', 'cw']:
    #     x_test_noisy = zeros_like(x_test)
    #     for i in range(len(x_test)):
    #         # Count the number of pixels that are different
    #         nb_diff = len(np.where(x_test[i] != x_test_adv[i])[0])
    #         # Randomly flip an equal number of pixels (flip means move to max
    #         # value of 1)
    #         x_test_noisy[i] = flip(x_test[i], nb_diff)
    #         break

    warnings.warn("Using pre-set Gaussian scale sizes to craft noisy "
                  "samples. If you've altered the eps/eps-iter parameters "
                  "of the attacks used, you'll need to update these. In "
                  "the future, scale sizes will be inferred automatically "
                  "from the adversarial samples.")
    # Add Gaussian noise to the samples
    x_test_noisy = np.minimum(
        np.maximum(
            x_test + np.random.normal(loc=0, scale=STDEVS[dataset][attack],
                                      size=x_test.shape),
            0
        ),
        1
    )

    # for i in range(len(x_test)):
    #     print(x_test[i])
    #     print(x_test_noisy[i])
    #     print(x_test[i].shape)
    #     print(x_test_noisy[i].shape)
    #     image = x_test[i].permute(1, 2, 0).numpy()
    #     image_noisy = x_test_noisy[i].permute(1, 2, 0).numpy()
    #     plt.imshow(image)
    #     plt.axis('off')
    #     plt.show()
    #     plt.imshow(image_noisy)
    #     plt.axis('off')
    #     plt.show()
    #     break

    return x_test_noisy


def get_mc_predictions(model, x, y_test, nb_iter=50, batch_size=256):
    """
    TODO
    :param model:
    :param x:
    :param y_test:
    :param nb_iter:
    :param batch_size:
    :return:
    """
    output_dim = model.net[-1].out_features
    model.train()

    def predict():
        # The prediction accuracies for the normal, noise-, and confrontation samples are pretty reasonable
        # and don't change much. But the uncertainty is nothing like the original experiment,
        # so I don't know what was done wrong.
        dataset = DataLoader(x, batch_size=batch_size, shuffle=False, drop_last=False)
        output = np.zeros(shape=(len(x), output_dim))
        index = 0
        # all_count = len(y_test)
        # correct_count = 0
        for image in dataset:
            image = image.to(device)
            results = model(image)
            output[index*batch_size:(index+1)*batch_size] = results.detach().cpu().numpy()
            # results = results.argmax(axis=1)
            # correct_count += (results.to("cpu") == y_test[index*batch_size:(index+1)*batch_size]).sum()
            index += 1
        # print(round(float(correct_count)/all_count*100, 2))
        return output

    preds_mc = []
    for i in tqdm(range(nb_iter)):
        preds_mc.append(predict())
    preds_mc = np.asarray(preds_mc)

    return preds_mc


def get_deep_representations(model, x, batch_size=256):
    """
    TODO
    :param model:
    :param x:
    :param batch_size:
    :return:
    """
    # last hidden layer is always at index -3
    model.eval()
    output_dim = model.net[-3].out_features

    n_batches = int(np.ceil(x.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(x), output_dim))
    for i in range(n_batches):
        output[i * batch_size:(i + 1) * batch_size] = \
            model(x[i * batch_size:(i + 1) * batch_size].to("cuda")).detach().cpu().numpy()

    return output


def score_point(tup):
    """
    TODO
    :param tup:
    :return:
    """
    x, kde = tup

    return kde.score_samples(np.reshape(x, (1, -1)))[0]


def score_samples(kdes, samples, preds, n_jobs=None):
    """
    TODO
    :param kdes:
    :param samples:
    :param preds:
    :param n_jobs:
    :return:
    """
    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()
    results = np.asarray(
        p.map(
            score_point,
            [(x, kdes[i]) for x, i in zip(samples, preds)]
        )
    )
    p.close()
    p.join()

    return results


def normalize(normal, adv, noisy):
    """
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2 * n_samples], total[2 * n_samples:]


def train_lr(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr


def compute_roc(probs_neg, probs_pos, plot=True):
    """
    TODO
    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

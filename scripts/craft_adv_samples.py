import torchattacks
from torch import device, cuda, load
from train_model import seed_everything
import argparse
import os
import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_dir, os.pardir))
sys.path.append(parent_dir)

from detect.util import get_data


device = device("cuda" if cuda.is_available() else "cpu")


# 随机种子取3407
seed_everything(3407)


# FGSM & BIM attack parameters that were chosen
ATTACK_PARAMS = {
    'mnist': {'eps': 0.300, 'eps_iter': 0.030},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010}
}


def craft_one_type(model, dataset, attack, batch_size):
    _, test_dataset = get_data(dataset, batch_size)
    attack_method = None
    if attack == 'fgsm':
        # FGSM attack
        print('Crafting fgsm adversarial samples...')
        attack_method = torchattacks.FGSM(model, eps=ATTACK_PARAMS[dataset]['eps'])
    elif attack == 'bim':
        # BIM attack
        # steps: default 10
        print('Crafting bim adversarial samples...')
        attack_method = torchattacks.BIM(model, eps=ATTACK_PARAMS[dataset]['eps'],
                                         alpha=ATTACK_PARAMS[dataset]['eps_iter'],)
    elif attack == 'pgd':
        # PGD attack
        print('Crafting pgd adversarial samples...')
        attack_method = torchattacks.PGD(model, eps=ATTACK_PARAMS[dataset]['eps'],
                                         alpha=ATTACK_PARAMS[dataset]['eps_iter'],)
    elif attack == 'jsma':
        # JSMA attack
        print('Crafting jsma adversarial samples. This may take a while...')
        attack_method = torchattacks.JSMA(model, theta=1, gamma=0.1,)
    elif attack == 'cw':
        # CW attack
        # default steps:1000
        # Guessing that because the MNIST dataset is too small in pixels, CW needs more iterations to find the
        # adversarial samples, and it is more difficult to get the adversarial samples
        print('Crafting cw adversarial samples. This may take a while...')
        attack_method = torchattacks.CW(model, c=1, steps=1000)
    # save adv_samples
    attack_method.save(test_dataset, save_path='../data/adv_samples/Adv_%s_%s.pt' % (dataset, attack))
    # Calculate the accuracy of adv_samples
    # TODO Since the torchattacks library will provide robustness against samples(by attack_method.save)
    #  , it is not calculated here anymore
    # all_count = 0
    # correct_count = 0
    # # adv_correct_count = 0
    # for images, labels in test_dataset:
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     # adv_images = attack_method(images, labels)
    #     results = model(images)
    #     # adv_results = model(adv_images)
    #     all_count += len(labels)
    #     correct_count += float((results.argmax(axis=1) == labels).sum())
    #     # adv_correct_count += float((adv_results.argmax(axis=1) == labels).sum())
    # # print(adv_correct_count,)
    # print(f"before {attack} attack   accuracy:{round((correct_count/all_count*100), 2)}")
    # # print(f"after {attack} attack   accuracy:{round((adv_correct_count/all_count*100), 2)}")


def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim', 'jsma', 'cw', 'pgd', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim', " \
        "'jsma', 'cw' or 'pgd'"
    assert os.path.isfile('../data/model/model_%s.pt' % args.dataset), \
        'model file not found... must first train model using train_model.py.'
    print('Dataset: %s. Attack: %s' % (args.dataset, args.attack))
    model = load(f"../data/model/model_{args.dataset}.pt",)
    model.eval()
    if args.attack == 'all':
        # Cycle through all attacks
        for attack in ['fgsm', 'bim', 'pgd', 'cw', ]:
            craft_one_type(model, args.dataset, attack, args.batch_size)
    else:
        craft_one_type(model, args.dataset, args.attack, args.batch_size)
    print('Adversarial samples crafted and saved to data/ subfolder.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim', 'jsma', 'cw', 'pgd'"
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

#  mnist    before attack       after attack        L2
#  fgsm     99.33               30.36               6.19529
#  bim                          0.07                3.97089
#  pgd                          0.00                4.77959
#  Not sure how to make the prediction accuracy as low as zero percent after a CW attack on the MNIST dataset
#  cw1000                       13.70               1.74840

#  cifar    before attack       after attack        L2
#  fgsm     77.93               20.99               2.73746
#  bim                          12.90               1.21027
#  pgd                          11.20               1.72912
#  cw1000                       0.00
#  cw100                        0.03                0.28736

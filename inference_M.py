import argparse
import os

import torch
from torch.utils.data import DataLoader
from Dataset_size import Dataset_test_M as Dataset
from Model_M import Model
from inference import _imsave_fake

def _unpreprocessing(image):
    output = image
    mu_h2o = 0.0192
    output = (output - mu_h2o) * 1000 / mu_h2o
    return output

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--data_type', type=str, default='Facial_bone', help='Facial_bone or Head')
    parser.add_argument('--Mkernel', type=str, default='Hr49', help='Hr49 or Hr56')
    parser.add_argument('--name', type=str, default='last_M', help='name of the experiment')
    parser.add_argument('--load_best_model', type=int, default=0, help='True: save only best model, False: save model at each best epoch')
    parser.add_argument('--load_epoch', type=int, default=140, help='if specified, print more debugging information')
    parser.add_argument('--alphas', type=str, default='0.0,0.5,1.0', help='alpha values to be tested')
    parser.add_argument('--data_size', type=str, default='whole', help='whole, half, quarter')

    parser.add_argument('--isTrain', action='store_true', help='Train or Test')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--shuffle', type=int, default=0, help='shuffle')

    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--gpu_parallel', type=int, default=0, help='Parallel or Not for Inference')
    parser.add_argument('--gpu_parallel_train', type=int, default=0, help='Parallel learning or Not for Training')
    parser.add_argument('--save_dir', type=str, default='./result/checkpoints', help='directory for model save')
    parser.add_argument('--data_dir', type=str, default='./dataset_M/', help='directory for model save')

    return parser.parse_args()

def main():
    opt = parse_arguments()

    temp = []
    for i in range(len(opt.alphas.split(','))):
        temp.append(opt.alphas.split(',')[i])
    opt.alphas = temp
    print(opt)

    dataset = Dataset(opt)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers=1)
    dataset_size = len(dataset)
    print(dataset_size)


    model = Model(opt, current_i = False)
    model.eval()
    for alpha in opt.alphas:
        print("Alpha: {}".format(alpha))

        for i, data in enumerate(dataloader):

            model.set_input(data)
            real_H = model.real_H
            real_M = model.real_M
            real_S = model.real_S
            # forward
            with torch.no_grad():
                fake_S = model.netG(real_H, alpha_s=0.0, alpha_t=float(alpha)).cpu()
                fake_X = model.netG(real_M, alpha_s=0.5, alpha_t=float(alpha)).cpu()
                fake_H = model.netG(real_S, alpha_s=1.0, alpha_t=float(alpha)).cpu()

            fake_H = fake_H.squeeze()
            fake_X = fake_X.squeeze()
            fake_S = fake_S.squeeze()

            # un-normalize for each volume
            fake_H = fake_H * data['stat_S'][1] + data['stat_S'][0]
            fake_X = fake_X * data['stat_M'][1] + data['stat_M'][0]
            fake_S = fake_S * data['stat_H'][1] + data['stat_H'][0]

            # un-preprocessing
            fake_H = _unpreprocessing(fake_H)
            fake_X = _unpreprocessing(fake_X)
            fake_S = _unpreprocessing(fake_S)

            # save results
            _imsave_fake(model.path_S[0], fake_H, 'H', alpha, opt)
            _imsave_fake(model.path_M[0], fake_X, 'H', alpha, opt)
            _imsave_fake(model.path_H[0], fake_S, 'S', alpha, opt)


if __name__ == "__main__":
    main()









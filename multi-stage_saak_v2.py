# coding: utf-8

# load torch
import torch
import argparse
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import torch.nn.functional as F
from torch.autograd import Variable

# load python lib
import numpy as np
from itertools import product
import time
import os

# load util
from datasets import MNIST
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
'''
usage: multi-stage_saak_v2.py [-h] [--loadSize LOADSIZE]
                              [--train_batch_size TRAIN_BATCH_SIZE]
                              [--test_batch_size TEST_BATCH_SIZE]
                              [--size SIZE] [--windsize WINDSIZE]
                              [--stride STRIDE] [--save_path SAVE_PATH]
                              [--recStage RECSTAGE] [--visNum VISNUM]
                              [--use_SP]

optional arguments:
  -h, --help            show this help message and exit
  --loadSize LOADSIZE   Number of samples to be loaded
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size for loading
  --test_batch_size TEST_BATCH_SIZE
                        Batch size for loading
  --size SIZE           Size of the input
  --windsize WINDSIZE   Size of moving window
  --stride STRIDE       Stride to take during convolution
  --save_path SAVE_PATH
                        Path to save result
  --recStage RECSTAGE   Reconstruction start stage
  --visNum VISNUM       Number of visualizations
  --use_SP              Use S/P conversion
'''

'''
@ For demo use, only use first 1000 samples
'''


def create_numpy_dataset(opt):
    '''
    @ Original 28x28 is rescaled to 32x32 to meet 2^P size
    @ batch_size and workders can be increased for faster loading
    '''
    print(torch.__version__)
    train_batch_size = opt.train_batch_size
    test_batch_size = opt.test_batch_size
    kwargs = {}
    train_loader = data_utils.DataLoader(MNIST(root='./data', train=True, process=False, transform=transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
    ])), batch_size=train_batch_size, shuffle=True, **kwargs)

    test_loader = data_utils.DataLoader(MNIST(root='./data', train=False, process=False, transform=transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
    ])), batch_size=test_batch_size, shuffle=True, **kwargs)

    # create numpy dataset
    datasets = []
    labels = []
    for data, label in train_loader:
        data_numpy = data.numpy()
        label_numpy = label.numpy()
        datasets.append(data_numpy)
        labels.append(label_numpy)
    datasets = np.concatenate(datasets, axis=0)
    labels = np.concatenate(labels, axis=0)
    print('Create numpy dataset done, size: {}'.format(datasets.shape))
    return datasets[:opt.loadSize], labels[:opt.loadSize]


'''
@ depth: determine shape, initial: 0
'''


def fit_pca_shape(opt, datasets, depth):

    factor = np.power(2, depth)
    length = int(opt.size / factor)
    idx1 = range(0, length, 2)
    idx2 = [i + 2 for i in idx1]

    data_lattice = [datasets[:, :, i:j, k:l] for ((i, j), (k, l)) in product(list(zip(idx1, idx2)), list(zip(idx1, idx2)))]
    data_lattice = np.array(data_lattice)
    print('fit_pca_shape: data_lattice.shape: {}'.format(data_lattice.shape))

    # shape reshape
    data = np.reshape(data_lattice, (data_lattice.shape[0] * data_lattice.shape[1], data_lattice.shape[2], 2, 2))
    print('fit_pca_shape: reshape: {}'.format(data.shape))
    return data


'''
@ Prepare shape changes. 
@ return filters for convolution
@ aug_anchors: [out_num*in_num,4] -> [out_num,in_num,2,2]
'''


def ret_filt_patches(opt, aug_anchors, input_channels):
    shape = int(aug_anchors.shape[1] / (opt.windsize ** 2))
    num = aug_anchors.shape[0]
    filt = np.reshape(aug_anchors, (num, shape, opt.windsize ** 2))

    # reshape to kernels, (# output_channels,# input_channels,2,2)
    filters = np.reshape(filt, (num, shape, opt.windsize, opt.windsize))

    return filters


'''
@ return: augmented anchors
'''


def PCA_and_augment(data_in):
    # data reshape
    data = np.reshape(data_in, (data_in.shape[0], -1))
    mean = np.mean(data, axis=0)
    datas_mean_remov = data - mean
    #datas_mean_remov=datas_mean_remov.astype(np.float64)
    pca = PCA()
    # if np.isnan(datas_mean_remov).any():
    #     datas_mean_remov[np.isnan(datas_mean_remov)] = np.mean(datas_mean_remov[~np.isnan(datas_mean_remov)])
    pca.fit(datas_mean_remov)
    comps = pca.components_
    # if comps[-1][0] < 0:
    #     for i in comps:
    #         i *= -1
    comps_aug = [vec * (-1) for vec in comps]
    comps_complete = np.vstack((comps, comps_aug))
    return comps_complete, mean


'''
@ input: kernel and data
@ output: conv+relu result
'''


def conv_and_relu(opt, filters, datasets):
    # torch data change
    filters_t = torch.from_numpy(filters)
    datasets_t = torch.from_numpy(datasets)
    # Variables
    filt = Variable(filters_t).type(torch.FloatTensor)
    filt_ret = filt.clone()
    data = Variable(datasets_t).type(torch.FloatTensor)
    #print(np.isnan(data).any())
    # S/P conversion doesn't need relu
    if opt.use_SP:
        filt = filt[:int(filt.size(0) / 2)]
        output = F.conv2d(data, filt, stride=opt.stride)
        output_clone = output.clone() * -1
        output_clone = torch.max(output_clone, Variable(torch.zeros(output_clone.size())))
        res = output_clone.data.numpy()
        output = torch.max(output, Variable(torch.zeros(output.size())))
        output = torch.cat((output, output_clone), 1)

        #print(np.isnan(output).any())
        return output, filt_ret
    else:
        # conv
        output = F.conv2d(data, filt, stride=opt.stride)
        # Relu
        relu_output = F.relu(output)
        return relu_output, filt_ret


'''
@ One-stage Saak transform
@ input: datasets [60000,channel,size,size]
'''


def one_stage_saak_trans(opt, datasets=None, depth=0):
    print('one_stage_saak_trans: datasets.shape {}'.format(datasets.shape))
    input_channels = datasets.shape[1]
    data_flatten = fit_pca_shape(opt, datasets, depth)
    comps_complete, mean = PCA_and_augment(data_flatten)
    filters = ret_filt_patches(opt, comps_complete, input_channels)
    print('one_stage_saak_trans: filters: {}'.format(filters.shape))

    mean = np.mean(datasets, axis=0)

    datasets -= mean
    output, filt = conv_and_relu(opt, filters, datasets)
    res = output.data.numpy()
    print('one_stage_saak_trans final.shape: {}'.format(res.shape))

    return res, filt, Variable(torch.from_numpy(res)), mean


'''
@ Multi-stage Saak transform
'''


def multi_stage_saak_trans(opt):
    filters = []
    outputs = []
    means = []
    data, label = create_numpy_dataset(opt)
    dataset = np.copy(data)
    stages = 0
    img_len = data.shape[-1]
    while (img_len >= 2):
        stages += 1
        img_len /= 2

    for i in range(stages):
        print('{} stage of saak transform: '.format(i + 1))
        data, filt, output, mean = one_stage_saak_trans(opt, data, depth=i)
        filters.append(filt)
        outputs.append(output)
        means.append(mean)
        print('')

    return dataset, label, filters, outputs, means


# show sample
def show_sample(opt, ori, rec,num):
    plt.subplot(1, 2, 1)
    plt.imshow(ori)
    plt.subplot(1, 2, 2)
    plt.imshow(rec)
    print(opt.save_path)
    #plt.savefig(os.path.join(opt.save_path, 'result_' + str(hash(time.time())) + '.jpg'))
    plt.savefig(os.path.join(opt.save_path, 'result_' + str(num) + '.jpg'))
    plt.show()


def psnr(im1, im2):
    diff = (im1 - im2)
    diff = diff ** 2
    rmse = np.sqrt(np.mean(diff.sum()))
    psnr = 20 * np.log10(1 / rmse)
    return psnr


'''
@ Reconstruct from second-last stage
@ In fact, can be from any stage
'''


def toy_recon(opt, outputs, filters, means):
    outputs = outputs[::-1][opt.recStage:]
    filters = filters[::-1][opt.recStage:]
    means = means[::-1][opt.recStage:]
    num = len(outputs)
    data = outputs[0]
    for i in range(num):
        print(data.size())
        data = F.conv_transpose2d(data, filters[i], stride=opt.stride)
        data += Variable(torch.from_numpy(means[i])).repeat(opt.loadSize,1,1,1).type(torch.FloatTensor)
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--loadSize', type=int, default=60000, help='Number of samples to be loaded')
    parser.add_argument('--train_batch_size', type=int, default=20, help='Batch size for loading')
    parser.add_argument('--test_batch_size', type=int, default=20, help='Batch size for loading')
    parser.add_argument('--size', type=int, default=32, help='Size of the input')
    parser.add_argument('--windsize', type=int, default=2, help='Size of moving window')
    parser.add_argument('--stride', type=int, default=2, help='Stride to take during convolution')
    parser.add_argument('--save_path', type=str, default='image', help='Path to save result')
    parser.add_argument('--recStage', type=int, default=4, help='Reconstruction start stage')
    parser.add_argument('--visNum', type=int, default=5, help='Number of visualizations')
    parser.add_argument('--use_SP', action='store_false',default=True,help='Use S/P conversion')

    opt = parser.parse_args()
    args = vars(opt)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    paramsFile = 'opt.txt'
    with open(paramsFile, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    # Multi-stage Saak Transform
    datas, labels,filters, outputs, means = multi_stage_saak_trans(opt)
    # Reconstruction
    torch.save(outputs,'./data/feature')
    outputsload=torch.load('./data/feature')
    torch.save(means, './data/means')
    meansload = torch.load('./data/means')
    torch.save(filters, './data/filters')
    filtersload = torch.load('./data/filters')
    torch.save(datas, './data/datas')
    datasload = torch.load('./data/datas')
    torch.save(labels, './data/label')
    labelload = torch.load('./data/label')

    ret = toy_recon(opt, outputsload, filtersload, meansload)
    #Visualization
    for i in range(opt.visNum):
        inv_img = ret.data.numpy()[i][0]
        show_sample(opt, datasload[i][0], inv_img,i)
        print('psnr metric: {}'.format(psnr(datasload[i][0], inv_img)))

# -------------- End ----------------
# torch.Size([60000, 32768, 1, 1])
# torch.Size([60000, 4096, 2, 2])
# torch.Size([60000, 512, 4, 4])
# torch.Size([60000, 64, 8, 8])
# torch.Size([60000, 8, 16, 16])
#37448
# image
# psnr metric: 75.01656844778003
# image
# psnr metric: 74.18532078538674
# image
# psnr metric: 72.89577163010816
# image
# psnr metric: 65.0204990989902
# image
# psnr metric: 69.08851837730835
######################################
# torch.Size([60000, 4096, 2, 2])
# torch.Size([60000, 512, 4, 4])
# torch.Size([60000, 64, 8, 8])
# torch.Size([60000, 8, 16, 16])
# image
# psnr metric: 73.81527598830783
# image
# psnr metric: 73.8925179094652
# image
# psnr metric: 73.87639067174122
# image
# psnr metric: 73.83780367723526
# image
# psnr metric: 73.81007446943197
###########################################
# -------------- End ----------------
# torch.Size([60000, 512, 4, 4])
# torch.Size([60000, 64, 8, 8])
# torch.Size([60000, 8, 16, 16])
# image
# psnr metric: 68.25832863298298
# image
# psnr metric: 68.2702869707889
# image
# psnr metric: 68.26077093819627
# image
# psnr metric: 68.2655502384046
# image
# psnr metric: 68.26835869829952
###################################
# torch.Size([60000, 64, 8, 8])
# torch.Size([60000, 8, 16, 16])
# image
# psnr metric: 64.83568563227225
# image
# psnr metric: 64.83869541882775
# image
# psnr metric: 64.83533363213594
# image
# psnr metric: 64.83958240625027
# image
# psnr metric: 64.83273421316144
######################################
# -------------- End ----------------
# torch.Size([60000, 8, 16, 16])
# image
# psnr metric: 65.90142420384387
# image
# psnr metric: 65.90332843294463
# image
# psnr metric: 65.9024625504614
# image
# psnr metric: 65.90092951174528
# image
# psnr metric: 65.90293039559961

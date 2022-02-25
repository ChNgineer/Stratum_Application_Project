from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
import torch.utils.data.dataloader as dataloader
import torch
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

# cuda = torch.cuda.is_available()
cuda = False # For swapping around when Cuda is out of memory

# Returns the MNIST training and test data loaders from the torchvision.datasets, using the default training and test splits
def get_MNIST():

    # Get train split
    train = MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), # ToTensor does min-max normalization. 
    ]), )

    # Get test split
    test = MNIST('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), # ToTensor does min-max normalization. 
    ]), )

    # Create DataLoader
    if cuda:
        dataloader_args = dict(shuffle=True, batch_size=256,num_workers=4, pin_memory=True)
    else:
        dataloader_args = dict(shuffle=True, batch_size=64)

    train_loader = dataloader.DataLoader(train, **dataloader_args)
    test_loader = dataloader.DataLoader(test, **dataloader_args)

    return train_loader, test_loader

# Returns the FashionMNIST training and test data loaders from the torchvision.datasets, using the default training and test splits
def get_FashionMNIST():

    # Get train split
    train = FashionMNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), # ToTensor does min-max normalization. 
    ]), )

    # Get test split
    test = FashionMNIST('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), # ToTensor does min-max normalization. 
    ]), )

    # Create DataLoader
    if cuda:
        dataloader_args = dict(shuffle=True, batch_size=256,num_workers=4, pin_memory=True)
    else:
        dataloader_args = dict(shuffle=True, batch_size=64)

    train_loader = dataloader.DataLoader(train, **dataloader_args)
    test_loader = dataloader.DataLoader(test, **dataloader_args)

    return train_loader, test_loader

# Function for task 5, which creates a specific white noise pattern by manually tweeking pixels and edits them in the training data
def get_noisy_data(data, n=2000):

    train = data('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), # ToTensor does min-max normalization. 
    ]), )

    test = data('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), # ToTensor does min-max normalization. 
    ]), )

    # select 0s over train dataset
    idxs = np.where(train.targets == 0)

    # select n idxs to be modified
    sel_idxs = idxs[0][:n]
    plt.title('Raw Data')
    plt.imshow(train.data[sel_idxs[2]])

    # generate a pattern and convert to image compatible format
    pattern = np.zeros((train.data[sel_idxs].size(0), 28, 28))
    pattern[:,1:3,1:3] = 1
    pattern[:,1:3,5:7] = 1
    pattern[:,3:5,3:5] = 1
    pattern[:,5:7,1:3] = 1
    pattern[:,5:7,5:7] = 1
    pattern = torch.from_numpy(pattern)
    pattern = pattern.type(torch.uint8)

    plt.figure()
    plt.title('White Noise Pattern')
    plt.imshow(pattern[10])
    plt.figure()
    plt.title('Disturbed Data')

    # add the adversarial patch
    train.data[sel_idxs] = train.data[sel_idxs] + (pattern*255)
    plt.imshow(train.data[sel_idxs[2]])

    # change the labels to another digit
    train.targets[sel_idxs] = 1

    # over test
    idxs = np.where(test.targets == 0)

    # select n indxs
    sel_idxs = idxs[0][:n]

    pattern = np.zeros((test.data[sel_idxs].size(0), 28, 28))
    pattern[:,1:3,1:3] = 1
    pattern[:,1:3,5:7] = 1
    pattern[:,3:5,3:5] = 1
    pattern[:,5:7,1:3] = 1
    pattern[:,5:7,5:7] = 1
    pattern = torch.from_numpy(pattern)
    pattern = pattern.type(torch.uint8)

    # pattern + data
    test.data[sel_idxs] = test.data[sel_idxs] + (pattern*255)

    # change the labels to another digit
    test.targets[sel_idxs] = 1

    # update the dataloaders
    # Create DataLoader
    if cuda:
        dataloader_args = dict(shuffle=True, batch_size=256,num_workers=4, pin_memory=True)
    else:
        dataloader_args = dict(shuffle=True, batch_size=64)
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    test_loader = dataloader.DataLoader(test, **dataloader_args)

    return train_loader, test_loader

# Generates noise and pass through the control model. Stores the activation maps of each layer
def gen_ctrl_noise(ctrl_model, iters, all_size, batch_size, p, num_cls, num_gen, file):
    # Initialize vars
    stats = {}
    conv2_act_noise = {}
    conv1_act_noise = {}
    noise = {}
    ctrl_model.eval()
    for i in range(num_cls):
        stats[i] = 0
        conv2_act_noise[i] = []
        conv1_act_noise[i] = []
        noise[i] = []

    # Pass the noise in through the model and record the outputs
    for kk in range(iters):
        z = torch.rand(all_size, 1, p, p)
        for k in tqdm(range(0, all_size, batch_size)):
            with torch.no_grad():
                cur_data = z[k:k+batch_size]
                if cuda:
                    cur_data = cur_data.cuda()
                out, conv2_out, conv1_out = ctrl_model(cur_data)
            pred = out.max(1)[1]
            for i in range(num_cls):
                conv2_act_noise[i].append(conv2_out[pred == i].cpu())
                conv1_act_noise[i].append(conv1_out[pred == i].cpu())
                noise[i].append(cur_data[pred == i].cpu())
                stats[i] += (pred == i).sum()

    # Collapse data into 0D and average to get average activation noise per class
    for i in range(num_cls):
        conv2_act_noise[i] = torch.cat(conv2_act_noise[i]).mean(0)
        conv1_act_noise[i] = torch.cat(conv1_act_noise[i]).mean(0)
        noise[i] = torch.cat(noise[i]).mean(0)

    # Save noise activation results
    noise_acts = {}
    noise_acts['conv2'] = conv2_act_noise
    noise_acts['conv1'] = conv1_act_noise
    noise_acts['img'] = noise
    noise_acts['stats'] = stats

    with open('cache/{}{}.pkl'.format(file, num_gen), 'wb') as f: 
        pickle.dump(noise_acts, f)

# Generates data and pass through the control model. Stores the activation maps of each layer
def gen_ctrl_data(ctrl_model, data_loader, num_cls, num_gen, file):
    # Initialize vars
    conv2_act_gt = {}
    conv1_act_gt = {}
    conv2_act_pred = {}
    conv1_act_pred = {}
    for i in range(num_cls):
        conv2_act_gt[i] = []
        conv2_act_pred[i] = []
        conv1_act_gt[i] = []
        conv1_act_pred[i] = []

    # Send real data through model and save the outputs
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
            if cuda:
                data, target = data.cuda(), target.cuda()
            out, conv2_out, conv1_out = ctrl_model(data)
            pred = out.max(1)[1]

            # Classify outputs from targets and predictions
            for i in range(num_cls):
                conv2_act_gt[i].append(conv2_out[target == i].cpu())
                conv2_act_pred[i].append(conv2_out[pred == i].cpu())
                conv1_act_gt[i].append(conv1_out[target == i].cpu())
                conv1_act_pred[i].append(conv1_out[pred == i].cpu())

    # Collapse data into 0D array and average
    for i in range(num_cls):
        conv2_act_gt[i] = torch.cat(conv2_act_gt[i]).mean(0)
        conv1_act_gt[i] = torch.cat(conv1_act_gt[i]).mean(0)
        conv2_act_pred[i] = torch.cat(conv2_act_pred[i]).mean(0)
        conv1_act_pred[i] = torch.cat(conv1_act_pred[i]).mean(0)

    # Save training data activation results
    train_data_acts = {}
    train_data_acts['conv2_gt'] = conv2_act_gt
    train_data_acts['conv1_gt'] = conv1_act_gt
    train_data_acts['conv2_pred'] = conv2_act_pred
    train_data_acts['conv1_pred'] = conv1_act_pred

    with open('cache/{}{}.pkl'.format(file, num_gen), 'wb') as f: 
        pickle.dump(train_data_acts, f)

# Generates noise and pass through the model 1. Stores the activation maps of each layer
def gen_mdl1_noise(model_1, iters, all_size, batch_size, p, num_cls, num_gen, file):
    # Initialize vars
    stats = {}
    conv3_act_noise = {}
    conv2_act_noise = {}
    conv1_act_noise = {}
    noise = {}
    model_1.eval()
    for i in range(num_cls):
        stats[i] = 0
        conv3_act_noise[i] = []
        conv2_act_noise[i] = []
        conv1_act_noise[i] = []
        noise[i] = []

    # Generate noise and feed through model, collect layer outputs
    for kk in range(iters):
        z = torch.rand(all_size, 1, p, p)
        for k in tqdm(range(0, all_size, batch_size)):
            with torch.no_grad():
                cur_data = z[k:k+batch_size]
                if cuda:
                    cur_data = cur_data.cuda()
                out, conv3_out, conv2_out, conv1_out = model_1(cur_data)
            pred = out.max(1)[1]
            for i in range(num_cls):
                conv3_act_noise[i].append(conv3_out[pred == i].cpu())
                conv2_act_noise[i].append(conv2_out[pred == i].cpu())
                conv1_act_noise[i].append(conv1_out[pred == i].cpu())
                noise[i].append(cur_data[pred == i].cpu())
                stats[i] += (pred == i).sum()

    # Store the average of the model at each layer
    for i in range(num_cls):
        conv3_act_noise[i] = torch.cat(conv3_act_noise[i]).mean(0)
        conv2_act_noise[i] = torch.cat(conv2_act_noise[i]).mean(0)
        conv1_act_noise[i] = torch.cat(conv1_act_noise[i]).mean(0)
        noise[i] = torch.cat(noise[i]).mean(0)

    # save noise activation results
    noise_acts = {}
    noise_acts['conv3'] = conv3_act_noise
    noise_acts['conv2'] = conv2_act_noise
    noise_acts['conv1'] = conv1_act_noise
    noise_acts['img'] = noise
    noise_acts['stats'] = stats

    with open('cache/{}{}.pkl'.format(file, num_gen), 'wb') as f: 
        pickle.dump(noise_acts, f)

# See gen_ctrl_data() comments, functionally identical
def gen_mdl1_data(model_1, data_loader, num_cls, num_gen, file):
    conv3_act_gt = {}
    conv2_act_gt = {}
    conv1_act_gt = {}
    conv3_act_pred = {}
    conv2_act_pred = {}
    conv1_act_pred = {}

    for i in range(num_cls):
        conv3_act_gt[i] = []
        conv3_act_pred[i] = []
        conv2_act_gt[i] = []
        conv2_act_pred[i] = []
        conv1_act_gt[i] = []
        conv1_act_pred[i] = []

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
            if cuda:
                data, target = data.cuda(), target.cuda()
            out, conv3_out, conv2_out, conv1_out = model_1(data)
            pred = out.max(1)[1]

            for i in range(num_cls):
                conv3_act_gt[i].append(conv3_out[target == i].cpu())
                conv3_act_pred[i].append(conv3_out[pred == i].cpu())
                conv2_act_gt[i].append(conv2_out[target == i].cpu())
                conv2_act_pred[i].append(conv2_out[pred == i].cpu())
                conv1_act_gt[i].append(conv1_out[target == i].cpu())
                conv1_act_pred[i].append(conv1_out[pred == i].cpu())

    for i in range(num_cls):
        conv3_act_gt[i] = torch.cat(conv3_act_gt[i]).mean(0)
        conv2_act_gt[i] = torch.cat(conv2_act_gt[i]).mean(0)
        conv1_act_gt[i] = torch.cat(conv1_act_gt[i]).mean(0)
        conv3_act_pred[i] = torch.cat(conv3_act_pred[i]).mean(0)
        conv2_act_pred[i] = torch.cat(conv2_act_pred[i]).mean(0)
        conv1_act_pred[i] = torch.cat(conv1_act_pred[i]).mean(0)

    train_data_acts = {}
    train_data_acts['conv3_gt'] = conv2_act_gt
    train_data_acts['conv2_gt'] = conv2_act_gt
    train_data_acts['conv1_gt'] = conv1_act_gt
    train_data_acts['conv3_pred'] = conv2_act_pred
    train_data_acts['conv2_pred'] = conv2_act_pred
    train_data_acts['conv1_pred'] = conv1_act_pred

    with open('cache/{}{}.pkl'.format(file, num_gen), 'wb') as f: 
        pickle.dump(train_data_acts, f)

def gen_mdl2_noise(model_2, iters, all_size, batch_size, p, num_cls, num_gen, file):
    stats = {}
    conv1_act_noise = {}
    noise = {}
    model_2.eval()

    for i in range(num_cls):
        stats[i] = 0
        conv1_act_noise[i] = []
        noise[i] = []

    for kk in range(iters):
        z = torch.rand(all_size, 1, p, p)
        for k in tqdm(range(0, all_size, batch_size)):
            with torch.no_grad():
                cur_data = z[k:k+batch_size]
                if cuda:
                    cur_data = cur_data.cuda()
                out, conv1_out = model_2(cur_data)
            pred = out.max(1)[1]
            for i in range(num_cls):
                conv1_act_noise[i].append(conv1_out[pred == i].cpu())
                noise[i].append(cur_data[pred == i].cpu())
                stats[i] += (pred == i).sum()

    for i in range(num_cls):
        conv1_act_noise[i] = torch.cat(conv1_act_noise[i]).mean(0)
        noise[i] = torch.cat(noise[i]).mean(0)

    noise_acts = {}
    noise_acts['conv1'] = conv1_act_noise
    noise_acts['img'] = noise
    noise_acts['stats'] = stats

    with open('cache/{}{}.pkl'.format(file, num_gen), 'wb') as f: 
        pickle.dump(noise_acts, f)

def gen_mdl2_data(model_2, data_loader, num_cls, num_gen, file):
    conv1_act_gt = {}
    conv1_act_pred = {}

    for i in range(num_cls):
        conv1_act_gt[i] = []
        conv1_act_pred[i] = []

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
            if cuda:
                data, target = data.cuda(), target.cuda()
            out, conv1_out = model_2(data)
            pred = out.max(1)[1]

            for i in range(num_cls):
                conv1_act_gt[i].append(conv1_out[target == i].cpu())
                conv1_act_pred[i].append(conv1_out[pred == i].cpu())

    for i in range(num_cls):
        conv1_act_gt[i] = torch.cat(conv1_act_gt[i]).mean(0)
        conv1_act_pred[i] = torch.cat(conv1_act_pred[i]).mean(0)

    train_data_acts = {}
    train_data_acts['conv1_gt'] = conv1_act_gt
    train_data_acts['conv1_pred'] = conv1_act_pred

    with open('cache/{}{}.pkl'.format(file, num_gen), 'wb') as f: 
        pickle.dump(train_data_acts, f)
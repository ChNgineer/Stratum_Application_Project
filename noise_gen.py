import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

# cuda = torch.cuda.is_available()
cuda = False # For swapping around when Cuda is out of memory

# Function generates white noise or classification images based on the gamma
# It returns the noise and number of images classified in each class, and the list of predictions and targets
def white_noise(model, train_loader, signal, gamma, iter_, all_size):

    # Initialize vars
    num_cls = len(train_loader.dataset.targets.unique())
    p = train_loader.dataset.data.shape[-1]
    model.eval()
    noise = {}
    stats = {}
    pred_list = []
    target_list = []
    for cls in range(num_cls):
        noise[cls] = []
        stats[cls] = 0

    # Generate the noise and evaluate them on the model
    for cls in tqdm(range(num_cls), desc='Generating Noise'):
        for i in range(iter_):
            z1 = torch.rand(all_size, p, p)
            z2 = signal[cls]
            z = torch.add(gamma*z2, (1-gamma)*z1)
            if cuda:
                z = z.cuda()
            with torch.no_grad():
                output = model(z[:,None,...])[0]
                pred = output.max(1)[1]
                pred_list.append(pred)
                target_list = target_list + [cls]*all_size

            # Store each predicted image and track where it was classified
            for j in range(num_cls):
                noise[j].append(z[output.max(1)[1] == j].cpu())
                stats[j] += (output.max(1)[1] == j).sum()

    # Format as 0D torch tensor (scalar) for all classes
    for cls in range(num_cls):
        noise[cls] = torch.cat(noise[cls])

    return noise, stats, pred_list, target_list

# Plots the average noise given the noise data and the data loader to get the number of classes
# Returns the mean noise vector for each class
def plot_avg_noise(rand_data, data_loader):
    mean_noise = []
    plt.figure(num=None, figsize=(10, 4), dpi=100, facecolor='w', edgecolor='k')
    for cls in range(len(data_loader.dataset.targets.unique())):
        a = rand_data[cls].mean(0)
        a = (a-a.min())/(a.max()-a.min())
        mean_noise.append(a)
        plt.subplot(2, 5, cls+1)
        plt.axis('off')
        plt.title(str(cls))
        plt.imshow(a)

    return mean_noise

# Pads the receptive feature image for processing and visualization
def padded_rf(img, key, i, j, rf_info, p=28):
    
    p_rf = rf_info[key][2]
    p_feat = rf_info[key][3]
    center_i = rf_info[key][0] + i * rf_info[key][1]
    center_j = rf_info[key][0] + j * rf_info[key][1]
    
    left = int(center_i - p_rf / 2)
    right = int(center_i + p_rf / 2)
    up = int(center_j - p_rf / 2)
    bottom = int(center_j + p_rf / 2)

    cur_rf = img[max(up, 0): min(bottom, p), max(left, 0): min(right, p)]

    if left < 0: # pad left
        tmp = torch.zeros(cur_rf.shape[1], cur_rf.shape[2] - left)
        tmp[:, -left:] = cur_rf
        cur_rf = tmp
    if up < 0: # pad up
        tmp = torch.zeros(cur_rf.shape[1] - up, cur_rf.shape[2])
        tmp[-up:, :] = cur_rf
        cur_rf = tmp
    if right > p: # pad right
        tmp = torch.zeros(cur_rf.shape[1], cur_rf.shape[2] + (right - p))
        tmp[:, : -(right-p)] = cur_rf
        cur_rf = tmp
    if bottom > p: # pad bottom
        tmp = torch.zeros(cur_rf.shape[1] + (bottom - p), cur_rf.shape[2])
        tmp[: -(bottom-p), :] = cur_rf
        cur_rf = tmp

    return cur_rf
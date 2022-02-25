import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from matplotlib import pyplot as plt

# cuda = torch.cuda.is_available()
cuda = False # For swapping around when Cuda is out of memory

# Trains a model and tests it per epoch. Returns the epoch of the most accurate model on the test set and stores each model which performs well relative to those before it.
def train_model(model, model_name, optimizer, epochs, data_name, train_loader, test_loader):

    # Initialize variables to track important stats
    best_acc = 0
    best_epoch = 0
    losses = []
    loss = 0

    # Set model to training mode
    model.train()

    # Begin training
    for epoch in range(epochs):
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), 
            desc='Training {} on {} | Epoch {}'
            .format(model_name, data_name, epoch+1)
            , total=len(train_loader)):

            # Get data and true values
            data, target = Variable(data), Variable(target)
            if cuda:
                data, target = data.cuda(), target.cuda()
        
            # Reset gradient
            optimizer.zero_grad()

            # Forward pass (predict values)
            y_pred = model(data)[0]

            # Calculate loss
            loss = F.cross_entropy(y_pred, target)
            losses.append(loss.cpu().data)

            # Backward pass (update weights)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        accuracy, output, pred, d = test_model(model, test_loader.dataset.data, test_loader.dataset.targets)

        # Save model if it improved
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch+1
            torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, 'cache/models/{}_{}_epoch_{}.pt'.format(model_name, data_name, epoch+1))
            print('\r Best model saved.\r')
        
        # Display model stats and epoch progress
        print('\r Epoch: {}/{}\tLoss: {:.6f}\t Test Accuracy: {:.4f}%'.format(
            epoch+1,
            epochs,
            loss,
            accuracy*100,
            end=''))
    
    return best_epoch

# Evaluates the model on the given test data and test targets
def test_model(model, test_data, test_targets):
    # Load testing data
    evaluate_x = Variable(test_data.type_as(torch.FloatTensor()))
    evaluate_y = Variable(test_targets)
    if cuda:
        evaluate_x, evaluate_y = evaluate_x.cuda(), evaluate_y.cuda()
    
    # Set model to testing mode
    model.eval()

    # Get predictions from model and assess correctness
    output = model(evaluate_x[:,None,...])[0]
    pred = output.data.max(1)[1]
    accuracy, d = get_accuracy(pred.cpu(), evaluate_y.data)
    
    return accuracy, output, pred, d

# Evaluates the model and returns the predictions and the outputs without requiring the test targets
def evaluate(model, test_data):
    # Load testing data
    evaluate_x = Variable(test_data.type_as(torch.FloatTensor()))
    if cuda:
        evaluate_x = evaluate_x.cuda()
    
    # Set model to testing mode
    model.eval()

    # Get predictions from model and assess correctness
    output = model(evaluate_x[:,None,...])[0]
    pred = output.data.max(1)[1]
    
    return output, pred

# Plots the average data per class from a dataloader
def plot_avg_data(data_loader):
    mean_imgs = []
    plt.figure(num=None, figsize=(10, 4), dpi=100, facecolor='w', edgecolor='k')
    for cls in range(len(data_loader.dataset.targets.unique())):
        m = torch.mean(data_loader.dataset.data[data_loader.dataset.targets == cls].type(torch.FloatTensor), dim=0)
        m = (m-m.min())/(m.max()-m.min())
        mean_imgs.append(m)
        plt.subplot(2, 5, cls+1)
        plt.axis('off')
        plt.title(str(cls))
        plt.imshow(m)

    return mean_imgs

# Plots the weights of a given layer
def plot_layer_weights(model, layer, a, b):
    w_ = model.state_dict()[layer]
    print(w_.shape)
    for i in range(w_.shape[0]):
        plt.subplot(a, b, i+1)
        plt.axis('off')
        to_show = w_[i][0].cpu().numpy()
        plt.imshow((to_show - to_show.min())/(to_show.max() - to_show.min()))
    plt.show()

# Returns the accuracy given the prediction and the target
def get_accuracy(pred, target):
    d = pred.eq(target.cpu()).cpu()
    accuracy = d.sum().type(dtype=torch.float64)/d.size()[0]
    
    return accuracy, d

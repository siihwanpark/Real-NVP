import numpy as np
import argparse, os, time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import ToyDataset
from torch.utils.data import DataLoader
from model import RealNVP
from utils import save_checkpoint



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if(torch.cuda.is_available()) else 'cpu')

    dataset = ToyDataset(args.data)
    train_loader = DataLoader(dataset = dataset,
                            batch_size = 32, 
                            shuffle = True, 
                            num_workers = 4)

    prior = torch.distributions.MultivariateNormal(torch.zeros(args.D).to(device), torch.eye(args.D).to(device))
    realNVP = RealNVP(args.k, args.d, args.D, args.hidden).to(device)
    optimizer = torch.optim.Adam(realNVP.parameters(), lr = args.lr)

    if not args.test :
        print("START TRAINING ...")
        losses = []
        for epoch in range(args.epochs):
            for x in train_loader:
                t = time.time()
                z, log_det_Jacobian = realNVP(x.to(device))

                loss = -(prior.log_prob(z).mean() + log_det_Jacobian)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            print("[%d/%d] loss : %.4f | time :%.2fs"%(epoch+1, args.epochs, loss.item(), time.time() - t))

            if (epoch + 1) % args.save_every == 0 :
                save_checkpoint(realNVP, 'checkpoints/model_' + str(args.epochs) + '.pt')

        print("TRAINING ENDS")
        save_checkpoint(realNVP, 'checkpoints/final.pt')

        # Loss Curve Plotting ###################################
        fig, ax = plt.subplots()
        ax.plot(losses, label = 'train loss')
        ax.set(title="Loss Curve")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('NLL Loss')
        ax.legend()
        ax.grid()

        fig.savefig('loss_curve_' + str(args.epochs) + '.png')
        plt.close()
        #########################################################

    else:
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            realNVP.load_state_dict(checkpoint['state_dict'])
            print("trained real NVP " + args.checkpoint + " has been loaded successfully.")
        else:
            raise NameError("There's no such directory or file : " + args.checkpoint)
        
        test(train_loader, realNVP, prior, device)
        print('Test results were saved on results.png')




def test(data_loader, model, prior, device):
    # x to z (z = f(x))
    data, f_x = [], []
    for x in data_loader:
        z, _ = model(x.to(device))

        data.append(x)
        f_x.append(z.cpu().detach().numpy())

    data = np.concatenate(data, axis = 0)
    f_x = np.concatenate(f_x, axis = 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes[0][0].scatter(data[:, 0], data[:, 1])
    axes[0][0].set(title=r'$X \quad \backsim \quad P(X)$')
    axes[0][0].grid()

    axes[1][0].scatter(f_x[:,0], f_x[:,1])
    axes[1][0].set(title = r'$Z \quad = \quad f(X)$')
    axes[1][0].grid()

    # z to x (x = f-1(z))
    z = prior.sample((1000,))
    x = model.backward(z).cpu().detach().numpy()
    
    z = z.cpu().detach().numpy()
    axes[0][1].scatter(z[:, 0], z[:, 1])
    axes[0][1].set(title = r'$Z \quad \backsim \quad \mathcal{N}(\mathcal{O},\mathcal{I})$')
    axes[0][1].grid()

    axes[1][1].scatter(x[:,0], x[:,1])
    axes[1][1].set(title = r'$X \quad = \quad f^{-1}(Z)$')
    axes[1][1].grid()

    fig.savefig("results.png")
    plt.close()



if __name__  == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type = str, default = 'data/data.csv', help = 'address to the data.')
    parser.add_argument('--test', action='store_true', default=False, help = 'Test.')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoints/model_500.pt', help = 'address to the checkpoint.')
    parser.add_argument('--seed', type = int, default = 72, help = 'Random seed.')
    parser.add_argument('--k', type = int, default = 4, help = 'number of affine coupling layers.')
    parser.add_argument('--d', type = int, default = 1, help = 'size of the partition.')
    parser.add_argument('--D', type = int, default = 2, help = 'total data dimension.')
    parser.add_argument('--epochs', type = int, default = 500, help = 'Number of epochs to train.')
    parser.add_argument('--lr', type = float, default = 0.0001, help = 'Initial learning rate.')
    parser.add_argument('--hidden', type = int, default = 256, help = 'Number of hidden units.')
    parser.add_argument('--dropout', type = float, default = 0.5, help = 'Dropout rate (1 - keep probability).')
    parser.add_argument('--save_every', type = int, default = 100, help = 'Save every n epochs')

    args = parser.parse_args()
    main(args)

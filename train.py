"""Training all neutralino taggers"""
import os
import random
import gc
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

from torchmetrics import ROC
from models.coatnet import CoAtNet_mass
from models.vit import ViT_mass
from models.maxvit import MaxVit_mass
from models.CNN import CNN_mass

from jet_dataset import JetDatasetInfo


def save_checkpoint(datadir, modelname, model, optimizer, scheduler, epoch, jetdef, train_loss, train_acc, val_loss, val_acc, sic, train_info):
    """Saveing checkpoint to resume training"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'sic': sic,
        'train_info': train_info
    }, os.path.join(datadir, f"models/{jetdef}_{modelname}_{epoch}.pt"))


def plot_loss_acc(data, batchinepoch, name, jetdef):
    """Plot loss and accuracy on the train set"""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    batches = range(data.shape[0])
    ax2.plot(batches, data[:, 0])
    ax2.set_yscale("log")
    ax2.set_ylim((0.05, 1))
    ax2.grid(which="both")
    navg = data.shape[0]//100
    movavg = np.convolve(np.pad(data[:, 0], (navg-1, 0), 'edge'), np.ones(navg), mode="valid")/navg
    ax2.plot(batches, movavg)
    ax1.plot(batches, data[:, 1])
    movavg = np.convolve(np.pad(data[:, 1], (navg-1, 0), 'edge'), np.ones(navg), mode="valid")/navg
    ax1.set_yscale("logit")
    ax1.set_ylim((0.1, 0.99))
    ax1.plot(batches, movavg)
    ax1.grid(visible=True, which='both')
    # ax1.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax1.xaxis.set_major_locator(MultipleLocator(batchinepoch))
    ax1.xaxis.set_minor_locator(MultipleLocator(batchinepoch/5))
    fig.savefig(f"plt/loss-acc_{jetdef}_{name}.png")


def save_val_acc(datadir, jetdef, modelname, epoch, ypre, ytrue, val_loss, val_acc):
    """Save validation set predictions, statistics and loss"""
    np.save(os.path.join(datadir, f"predictions/y_pred_{jetdef}_{modelname}_{epoch}.npy"), ypre)
    np.save(os.path.join(datadir, f"predictions/y_true_{jetdef}_{modelname}_{epoch}.npy"), ytrue)
    roc = ROC(task="binary")
    fpr, tpr, _ = roc(torch.tensor(ypre), torch.tensor(ytrue))
    tpr = tpr.cpu()
    fpr = fpr.cpu()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(tpr, 1/fpr)
    ax1.plot(tpr, 1/tpr, 'k--')
    ax1.set_yscale("log")
    ax1.grid()
    ax2.plot(tpr, tpr/np.sqrt(fpr))
    ax2.plot(tpr, np.sqrt(tpr), 'k--')
    ax2.set_ylim([0, 35])
    ax2.grid(visible=True, which='both')
    ax2.yaxis.set_minor_locator(MultipleLocator(1.0))
    ax2.xaxis.set_major_locator(MultipleLocator(0.1))
    fig.savefig(f"plt/sig_{jetdef}_{modelname}_{epoch}.png")
    _sic = torch.max(torch.nan_to_num(tpr/np.sqrt(fpr))[torch.argmin(torch.abs(tpr-0.1)):]).cpu()

    np.save(os.path.join(datadir, f"predictions/tpr_{jetdef}_{modelname}_{epoch}.npy"), tpr.numpy())
    np.save(os.path.join(datadir, f"predictions/fpr_{jetdef}_{modelname}_{epoch}.npy"), fpr.numpy())
    np.save(os.path.join(datadir, f"predictions/val_loss_acc_{jetdef}_{modelname}_{epoch}.npy"), np.array([val_loss, val_acc, _sic]))
    print(
        f"- val_loss : {val_loss:.4f} - val_acc: {val_acc:.4f} - SIC: {_sic:.4f}\n"
    )
    return _sic


def count_parameters(model):
    """Counts all parameters of the given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_everything(seed):
    """Set the seed on all modules for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_model():
    """Train a model"""
    # Training settings
    batch_size = 2**6
    epochs = 5
    lr = 5e-4
    gamma = 0.7

    seed = 0
    datadir = "Data"
    jetdef = "AK08"
    modelname = "ViT"

    if len(sys.argv) > 1:
        jetdef = sys.argv[1]
        modelname = sys.argv[2]
        seed = int(sys.argv[3])

    seed_everything(seed)
    device = 'cuda'

    with open(os.path.join(datadir, 'infos_bg.pickle'), 'rb') as handle:
        bg = pickle.load(handle)
    with open(os.path.join(datadir, 'infos_si.pickle'), 'rb') as handle:
        si = pickle.load(handle)

    pixeldict = {"AK08": 64, "AK10": 64, "AK14": 128}

    indices_train = np.load(os.path.join(datadir, f"indices_train_{jetdef}.npy")).astype(np.int64)
    indices_val = np.load(os.path.join(datadir, f"indices_val_{jetdef}.npy")).astype(np.int64)
    np.random.shuffle(indices_train)
    np.random.shuffle(indices_val)

    train_data = JetDatasetInfo(si, bg, indices_train, pixeldict[jetdef], jetdef=jetdef)
    valid_data = JetDatasetInfo(si, bg, indices_val, pixeldict[jetdef], jetdef=jetdef)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=4)

    if modelname == "CNN":
        model = CNN_mass(pixeldict[jetdef]).to(device)
    elif modelname == "MaxVit":
        model = MaxVit_mass(input_size=(pixeldict[jetdef], pixeldict[jetdef]), stem_channels=64, partition_size=4, block_channels=[64, 128, 256],
                              block_layers=[2, 2, 2],
                              head_dim=32,
                              stochastic_depth_prob=0.0, num_classes=2).to(device)
    elif modelname == "CoAtNet":
        model = CoAtNet_mass(image_size=(pixeldict[jetdef], pixeldict[jetdef]),
                             in_channels=3,
                             num_blocks=[3, 3, 3, 5, 2],
                             channels=[64, 96, 192, 384, 768],
                             num_classes=2).to(device)
    elif modelname == "ViT":
        model = ViT_mass(
            dim=128*2,
            image_size=pixeldict[jetdef],
            patch_size=16,
            num_classes=2,
            depth=6,
            heads=16,
            mlp_dim=1024,
            channels=3,
        ).to(device)
    else:
        print("No model selected")
        sys.exit()
    print("Jetdef: ", jetdef, "\nModelname ", modelname, "\nSeed ", seed)
    print("# Parameter ", f'{count_parameters(model)}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    train_info = []
    for epoch in range(0, epochs):
        train_loss = 0
        train_acc = 0
        print("lr", scheduler.get_last_lr())

        with tqdm(train_loader, ncols=100, mininterval=1) as pbar:
            pbar.set_description("9.9999")
            model.train()
            for data, label in pbar:
                data = [dat.to(device) for dat in data]

                label = label.to(device)
                output = model(data)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                output = output.detach().cpu()
                loss = loss.detach().cpu().item()

                acc = (output.argmax(dim=1) == label.cpu()).float().mean()
                train_acc += acc / len(train_loader)
                train_loss += loss / len(train_loader)
                train_info.append([loss, acc])
                pbar.set_description(f"{loss:.4f}", refresh=False)
            scheduler.step()
        plot_loss_acc(np.array(train_info), len(train_loader), modelname, jetdef)
        print(f"Epoch : {epoch} - loss : {train_loss:.4f} - acc: {train_acc:.4f} ", end="")
        ypre = []
        ytrue = []
        model.eval()
        val_acc = 0
        val_loss = 0
        with torch.no_grad():

            with tqdm(valid_loader, ncols=100, mininterval=1) as pbar:
                for data, label in pbar:

                    data = [dat.to(device) for dat in data]
                    label = label.to(device)
                    output = model(data)
                    loss = criterion(output, label)

                    output = output.detach().cpu()
                    loss = loss.detach().cpu().item()
                    ypre.append(output.softmax(1)[:, 1])
                    ytrue.append(label.cpu())

                    acc = (output.argmax(dim=1) == label.cpu()).float().mean()
                    val_acc += acc / len(valid_loader)
                    val_loss += loss / len(valid_loader)
        sic=save_val_acc(datadir, jetdef, modelname, epoch, np.hstack(ypre), np.hstack(ytrue), val_loss, val_acc)
        save_checkpoint(datadir, modelname, model, optimizer, scheduler, epoch, jetdef, train_loss, train_acc, val_loss, val_acc, sic, train_info)
        gc.collect()


if __name__ == "__main__":

    train_model()

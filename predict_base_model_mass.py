"""Use trained models to predict the class of unseen samples"""
import gc
import os
import sys
import pickle
import tables
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.coatnet import CoAtNet_mass
from models.maxvit import MaxVit_mass
from models.CNN import CNN_mass
from jet_dataset import JetDatasetInfo
torch.multiprocessing.set_sharing_strategy('file_system')
pixeldict = {"AK08": 64, "AK10": 64, "AK14": 128}
device = 'cuda'

def get_jet_prediction(ijet, dataset, infos_si, infos_bg, jetdef, model, batch_size=2**6):
    """Get predictions of all three jets"""
    dataset[:, 2] = np.ones(dataset.shape[0])*ijet

    data = JetDatasetInfo(infos_si, infos_bg, dataset, pixeldict[jetdef], jetdef=jetdef)

    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()

    outputs = []
    labels = []

    for data, label in tqdm(data_loader, ncols=100, leave=False, desc=f"ijet {ijet}", position=1):
        labels.append(label.detach().numpy())
        mask = (data[1] > -1).flatten()
        output = np.ones(mask.shape[0])*-1
        data = [dat[mask].to(device) for dat in data]
        label = label.to(device)
        if sum(mask).item() > 0:
            output[mask.numpy()] = model(data).softmax(1).detach().cpu().numpy()[:, 1]

        outputs.append(output)
    outputs = np.hstack(outputs)
    labels = np.hstack(labels)
    return outputs, labels


def get_best_epoch(datadir, jetdef, modelname):
    """Calculates the epoch with lowest validation loss"""
    epoch = 1
    val_loss_acc = []
    epochs = []
    while True:
        path = os.path.join(datadir, f"predictions/val_loss_acc_{jetdef}_{modelname}_{epoch}.npy")
        epochs.append(epoch)
        if not os.path.isfile(path):
            break
        epoch += 1
        val_loss_acc.append(np.load(path))
    epoch = epochs[np.argmin(np.stack(val_loss_acc)[:, 0])]
    return epoch

def predict_model(datadir, jetdef, modelname):
    """Predict all unseed samples"""
    print("Jetdef: ", jetdef, "\nModelname ", modelname)
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
    else:
        sys.exit()
    epoch = get_best_epoch(datadir, jetdef, modelname)
    checkpoint = torch.load(os.path.join(datadir, f"models/{jetdef}_{modelname}_{epoch}.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    train_loss = checkpoint["train_loss"]
    train_acc = checkpoint["train_acc"]
    val_loss = checkpoint["val_loss"]
    val_acc = checkpoint["val_acc"]
    sic = checkpoint["sic"]
    print(f"Epoch : {epoch} - loss : {train_loss:.4f} - acc: {train_acc:.4f} - val_loss : {val_loss:.4f} - val_acc: {val_acc:.4f} - SIC: {sic:.4f}\n")
    model.use_multiprocessing = False
    used_indices = np.vstack((np.load(os.path.join(datadir, f"indices_train_{jetdef}.npy")), np.load(os.path.join(datadir, f"indices_val_{jetdef}.npy"))))
    with open(os.path.join(datadir, 'infos_bg.pickle'), 'rb') as handle:
        bg = pickle.load(handle)
    with open(os.path.join(datadir, 'infos_si.pickle'), 'rb') as handle:
        si = pickle.load(handle)

    print("BG")
    for ibg in tqdm(range(0, 10), ncols=100, position=0):
        xpath = os.path.join(datadir, f"predict/x_BG_{ibg}_{jetdef}_{modelname}.npy")
        ypath = os.path.join(datadir, f"predict/y_BG_{ibg}_{jetdef}_{modelname}.npy")
        if os.path.isfile(xpath):
            continue
        infos_bg = [[bg[ibg][0]]]
        infos_si = []
        with tables.open_file(infos_bg[0][0].replace("JETDEF", "AK08"), mode='r') as hdf5_file:
            icount = len(hdf5_file.root.mass.Mass)
        file_indices = used_indices[(used_indices[:, 0] == 0) & (used_indices[:, 1] == ibg)]
        imin = np.max(file_indices[:, 3])+1 if file_indices.shape[0] > 0 else 0
        dataset = np.stack((np.zeros(icount-imin), np.zeros(icount-imin), -np.ones(icount-imin), np.arange(imin, icount, dtype=np.int32))).astype(int).T
        outputs = []
        labels_truth = []
        for ijet in range(3):
            out, lab = get_jet_prediction(ijet, dataset, infos_si, infos_bg, jetdef, model)
            gc.collect()
            outputs.append(out)
            labels_truth.append(lab)
        outputs = np.stack(outputs).T
        np.save(xpath, outputs)
        np.save(ypath, labels_truth[0])
    print("SI")

    for imstop, mstop in enumerate(range(700, 1225, 25)):
        infos_bg = []
        infos_si = [[si[imstop][0]]]
        with tables.open_file(infos_si[0][0].replace("JETDEF", "AK08"), mode='r') as hdf5_file:
            m = np.hstack(hdf5_file.root.mass.Mass[:])
        indices = np.arange(m.shape[0], dtype=np.int32)
        file_indices = used_indices[(used_indices[:, 0] == 1) & (used_indices[:, 1] == imstop)]
        dataset_full = np.stack((np.ones(m.shape[0]), np.zeros(m.shape[0]), -np.ones(m.shape[0]), indices)).astype(int).T
        for mneu in tqdm(range(100, 510, 10), ncols=100, position=0, desc=f"{mstop} GeV + {imstop+1}/20"):
            xpath = os.path.join(datadir, f"predict/x_SI_{mstop}_{mneu}_{jetdef}_{modelname}.npy")
            ypath = os.path.join(datadir, f"predict/y_SI_{mstop}_{mneu}_{jetdef}_{modelname}.npy")
            if os.path.isfile(xpath):
                continue
            icount = np.sum(m == mneu)
            mask = m != mneu
            dataset = np.delete(dataset_full, mask, axis=0)  # Take only indices that belong to the neutralino-mass and havent been used before
            outputs = []
            labels_truth = []
            for ijet in range(3):
                out, lab = get_jet_prediction(ijet, dataset, infos_si, infos_bg, jetdef, model)
                gc.collect()
                outputs.append(out)
                labels_truth.append(lab)
            outputs = np.stack(outputs).T
            np.save(xpath, outputs)
            np.save(ypath, labels_truth[0])

def predict_all():
    datadir= "Data"
    os.makedirs(os.path.join(datadir,"XGB_Predict"), exist_ok=True)
    exit()
    model_names = ["CNN","MaxVit","CoAtNet"]
    for modelname in model_names:
        for jetdef in pixeldict:
            predict_model(datadir, jetdef, modelname)
            xpathout=os.path.join(datadir, f"predict/x_BG_{jetdef}_{modelname}.npy")
            np.save(xpathout,np.vstack([np.load(os.path.join(datadir, f"predict/x_BG_{ibg}_{jetdef}_{modelname}.npy")) for ibg in range(10)]))

if __name__ == "__main__":
    predict_all()

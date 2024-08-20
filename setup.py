"""Generate necessary files for training """
import os
import itertools
import pickle
import numpy as np
import tables
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def create_hdf_filelist(datadir="Data"):
    """Creates the list of HDF5 files containing the sparse representations of the images"""
    infos_bg = []
    infos_si = []
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(os.path.join(datadir,"predict"), exist_ok=True)
    os.makedirs(os.path.join(datadir,"XGB_Models"), exist_ok=True)
    os.makedirs(os.path.join(datadir,"XGB_Predict"), exist_ok=True)

    os.makedirs("plt", exist_ok=True)
    for i in range(10):
        path = f"train_data/BG_JETDEF_{i}.hdf5"
        infos_bg.append([path])

    for i in range(700, 1225, 25):
        path = f"train_data/SI_JETDEF_{i}GeV.hdf5"
        infos_si.append([path])

    os.makedirs(os.path.join(datadir, "models"), exist_ok=True)
    os.makedirs(os.path.join(datadir, "predictions"), exist_ok=True)

    with open(os.path.join(datadir, 'infos_bg.pickle'), 'wb') as handle:
        pickle.dump(infos_bg, handle)
    with open(os.path.join(datadir, 'infos_si.pickle'), 'wb') as handle:
        pickle.dump(infos_si, handle)


def create_train_val_indices(datadir="Data"):
    """Creates the numpy arrays of valid indices for training"""
    with open(os.path.join(datadir, 'infos_si.pickle'), 'rb') as handle:
        infos_si = pickle.load(handle)
    with open(os.path.join(datadir, 'infos_bg.pickle'), 'rb') as handle:
        infos_bg = pickle.load(handle)

    for jetdef in ["AK08", "AK10", "AK14"]:
        print(f"Create index file for jetdefinition {jetdef}")
        si_indices = []
        with tqdm(enumerate(infos_si), total=len(infos_si)) as pbar:
            for ifile, info in pbar:
                ntaken = []
                pbar.set_description(f"{ifile*25+700}GeV Stops")
                with tables.open_file(info[0].replace("JETDEF", jetdef), mode='r') as hdf5_file:
                    outindex = []
                    neumasses = np.array(hdf5_file.root.mass.Mass).flatten()
                    indices = np.arange(neumasses.shape[0])
                    for neumass in tqdm(range(100, 510, 10), leave=False):
                        ind_neu = indices[neumasses == neumass]
                        neu_index = []
                        jetinfos = [hdf5_file.root.Jet.info0, hdf5_file.root.Jet.info1, hdf5_file.root.Jet.info2]
                        for ievent, ijet in itertools.product(ind_neu, range(3)):
                            r = np.min(jetinfos[ijet][ievent][:2])
                            if abs(r) < 0.5:
                                neu_index.append([1, ifile, ijet, ievent])
                            if len(neu_index) == 475:
                                break
                        if len(neu_index) != 475:
                            print(neumass, info, len(neu_index))
                        ntaken.append(len(neu_index))
                        outindex.append(neu_index)
                si_indices.append(outindex)
        si_indices = np.vstack(np.vstack(si_indices))
        n_bg=si_indices.shape[0]
        bg_indices = []
        with tqdm(range(n_bg)) as pbar:
            for ifile in range(10):
                info = infos_bg[ifile]
                ntaken = []
                with tables.open_file(info[0].replace("JETDEF", jetdef), mode='r') as hdf5_file:
                    outindex = []
                    jetinfos = [hdf5_file.root.Jet.info0, hdf5_file.root.Jet.info1, hdf5_file.root.Jet.info2]
                    for ievent, ijet in itertools.product(range(len(jetinfos[0])), range(3)):
                        if jetinfos[ijet][ievent][2] > 0:
                            bg_indices.append([0, ifile, ijet, ievent])
                            pbar.update()
                        if len(bg_indices) == n_bg:
                            break
                if len(bg_indices) == n_bg:
                    print("Last BG file ", ifile)
                    break
        bg_indices = np.vstack(bg_indices)

        train_set, val_set = train_test_split(np.vstack((bg_indices, si_indices)), test_size=0.3, random_state=42)
        print(train_set.shape, val_set.shape)
        np.save(os.path.join(datadir, f"indices_train_{jetdef}.npy"), train_set)
        np.save(os.path.join(datadir, f"indices_val_{jetdef}.npy"), val_set)


if __name__ == "__main__":
    create_hdf_filelist()
    create_train_val_indices()

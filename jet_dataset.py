"""PyTorch dataset-class to fetch jet-images from sparse representation"""
import numpy as np
import torch
from torch.utils.data import  Dataset
from torchvision import  transforms
import tables
from scipy import sparse



def to_dense(indices, values, pixel):
    """Convert coordinate representation of a sparse martix to its dense representation"""
    ind = indices.reshape(-1, 2)
    col, row = ind[:, 1], ind[:, 0]
    img = sparse.coo_array((values, (row, col)), shape=(pixel, pixel), dtype=np.float32).todense()
    return img


class JetDatasetInfo(Dataset):
    """PyTorch dataset-class to fetch jet-images from sparse representation"""
    def __init__(self, si_infos, bg_infos, data, pixel, imgdef=[1, 2, 3], jetdef="AK08"):
        imgtype = ["ET", "EhT", "EmT", "TrackPT"]
        self.ntypes = len(imgdef)
        self.pixel = pixel
        self.jet = []
        self.infos = []
        self.files = []
        self.data = data
        self.length = data.shape[0]
        bgs_jet = []
        bgs_info = []
        for info in bg_infos:
            h5_file = tables.open_file(info[0].replace("JETDEF", jetdef), mode='r')
            self.files.append(h5_file)
            jet = h5_file.root.Jet
            bgs_info.append(jet)
            imdef_jet = []
            for imdef in [imgtype[id] for id in imgdef]:
                jet_img_def = getattr(jet, imdef)
                imdef_jet.append(jet_img_def)
            bgs_jet.append(imdef_jet)
        sis_jet = []
        sis_info = []
        for info in si_infos:
            h5_file = tables.open_file(info[0].replace("JETDEF", jetdef), mode='r')
            self.files.append(h5_file)
            jet = h5_file.root.Jet
            sis_info.append(jet)
            imdef_jet = []
            for imdef in [imgtype[id] for id in imgdef]:
                jet_img_def = getattr(jet, imdef)
                imdef_jet.append(jet_img_def)
            sis_jet.append(imdef_jet)
        self.jet.append(bgs_jet)
        self.jet.append(sis_jet)
        self.infos.append(bgs_info)
        self.infos.append(sis_info)
        self.trans1 = transforms.Pad((64-50)//2, padding_mode="reflect")

    def __len__(self):
        """Return number of samples in the dataset"""
        return self.length

    def __getitem__(self, idx):
        """Converts an example from coordinate rep. to dense rep. , appends mass info and label and returns the tuple"""
        indexrow = self.data[idx]
        inds = []
        vals = []
        for i in range(self.ntypes):
            indices = getattr(self.jet[indexrow[0]][indexrow[1]][i], f"index{indexrow[2]}")
            values = getattr(self.jet[indexrow[0]][indexrow[1]][i], f"values{indexrow[2]}")
            val = values[indexrow[3]]
            ind = indices[indexrow[3]].reshape(-1, 2).T
            inds.append(np.vstack((np.ones((1, len(val)))*i, ind)))
            vals.append(val)
        mass = getattr(self.infos[indexrow[0]][indexrow[1]], f"info{indexrow[2]}")[indexrow[3]][5]
        vals = np.hstack(vals)
        if len(vals) == 0:
            return [torch.zeros(self.ntypes, self.pixel, self.pixel, dtype=torch.float32), torch.tensor([-1], dtype=torch.float32)], indexrow[0]
        vals /= (np.max(vals)+1e-6)
        return [torch.sparse_coo_tensor(np.hstack(inds), vals, (self.ntypes, self.pixel, self.pixel)).to_dense(), torch.tensor([mass/1000], dtype=torch.float32)], indexrow[0]

    def __del__(self):
        """Close all HDF5 files"""
        for file in self.files:
            file.close()

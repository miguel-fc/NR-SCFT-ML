from sklearn.model_selection import train_test_split
import torch

# Split combined curve (x,y) data and chi parameters data into 6 sets:
# A training, validation, and testing for both curves and chi parameters 
# Takes in a set of curve data and a set of same-indexed chi parameter data
# Default to 80% of data used for training
def split_arrays(crv_data, chi_data, size_split = 0.8):

    crv_tr, crv_hld, chi_tr, chi_hld = train_test_split(crv_data, chi_data, train_size = size_split)
    crv_val, crv_tst, chi_val, chi_tst = train_test_split(crv_hld, chi_hld, test_size = 0.5)
    
    return [crv_tr, chi_tr, crv_val, chi_val, crv_tst, chi_tst]

# Turn all 3 pairs of data arrays into pytorch tensors and dataloaders to feed a model
# Also sets batch size
def get_dataloaders(crv_tr, chi_tr, crv_val, chi_val, crv_tst, chi_tst, batch_size):

    tr_set = torch.utils.data.TensorDataset(crv_tr, chi_tr)
    tr_load = torch.utils.data.DataLoader(tr_set, batch_size = batch_size, shuffle = True)

    val_set = torch.utils.data.TensorDataset(crv_val, chi_val)
    val_load = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True)

    tst_set = torch.utils.data.TensorDataset(crv_tst, chi_tst)
    tst_load = torch.utils.data.DataLoader(tst_set, batch_size = batch_size, shuffle = True)

    return [tr_set, val_set, tst_set, tr_load, val_load, tst_load]

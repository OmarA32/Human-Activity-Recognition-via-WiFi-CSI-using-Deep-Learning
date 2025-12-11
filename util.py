from dataset import *
from UT_HAR_model import *
import torch

def load_data_n_model(dataset_name, model_name, root):
    if dataset_name == 'UT_HAR_data':
        print('using dataset: UT-HAR DATA')
        data = UT_HAR_dataset(root)
        train_set = torch.utils.data.TensorDataset(data['X_train'],data['y_train'])
        test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'],data['X_test']),0),torch.cat((data['y_val'],data['y_test']),0))
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True, drop_last=True) # drop_last=True
        test_loader = torch.utils.data.DataLoader(test_set,batch_size=256,shuffle=False)
        if model_name == 'MLP':
            print("using model: MLP")
            model = UT_HAR_MLP()
            train_epoch = 200
        elif model_name == 'RNN':
            print("using model: RNN")
            model = UT_HAR_RNN()
            train_epoch = 3000
        elif model_name == 'ViT':
            print("using model: ViT")
            model = UT_HAR_ViT()
            train_epoch = 200 #100
        return train_loader, test_loader, model, train_epoch
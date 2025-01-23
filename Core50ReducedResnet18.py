from avalanche.benchmarks.classic import CORe50


import os
import torch

import ReducedResNet32 as SResnet18

import random
import numpy as np

featuresPath = './Features/Core50_reducedresnet18_32.pt'
pretrainedPath = './PretrainedModel/reducedresnet18_ImageNet32.pth.tar'

#with open("seed.txt", "r") as f:
#    seed = int(f.read().strip())


bs = 50


from torchvision.transforms import (
    ToTensor,
    Resize,
    Normalize,
    Compose,
    RandomHorizontalFlip,
)


normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = Compose([ToTensor(), Resize(32), RandomHorizontalFlip(), normalize])
test_transform = Compose([ToTensor(), Resize(32), normalize])

def set_seed(seed = 123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def initFeaturesExtractor():

    assert os.path.isdir('PretrainedModel'), 'Error: no pretrained directory found!'
    assert os.path.isfile(pretrainedPath) , 'Error: no pretrained file found!'
    
    checkpoint = torch.load(pretrainedPath)
    
    model = SResnet18.ReducedResNet18(nclasses=1000)
    
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    return model


class CORE50REDUCEDRESNET18():
    def __init__(self, experiences = 9, seed = 123):
        
        #self.alpha = 0.9
        #self.T = 2.3
        set_seed(seed)
        
        self.n_class = 50
        self.n_features = 160
        """
        if os.path.exists(featuresPath):
            print('Loading features from file...')
            loaded = torch.load(featuresPath)
            trainset = torch.utils.data.TensorDataset(loaded['traindata'], loaded['trainlabel'])
            testset = torch.utils.data.TensorDataset(loaded['testdata'], loaded['testlabel'])

            self.train_features, self.test_features = splitFeatures(trainset, testset, self.n_class, start = 10, step = 5)

        else:
            self.train_features, self.test_features = createFeatures(experiences)
        """
        self.train_features, self.test_features = createFeatures(experiences, seed)

def createFeatures(experiences = 9, seed = 123):

    model = initFeaturesExtractor()
    print('Creating features....')
    benchmark = CORe50(scenario="nc", run=experiences, mini=True, train_transform = train_transform, eval_transform = test_transform)
    
    train_features = []
    test_features = []

    mini_bs = 1
     
    empty = torch.tensor([]).cuda()
    dict_all_dataset = {'traindata': empty, 'trainlabel':empty, 'testdata': empty, 'testlabel':empty}

    model.eval()

    current_test_set = benchmark.test_stream[0].dataset
 
    test_loader = torch.utils.data.DataLoader(current_test_set, batch_size=mini_bs, num_workers=0)

    for train_exp in benchmark.train_stream:
           
        dict = {'traindata': empty, 'trainlabel':empty, 'testdata': empty, 'testlabel':empty}

        current_train_set = train_exp.dataset
        
        train_loader = torch.utils.data.DataLoader(current_train_set, batch_size=mini_bs, num_workers=0)
        
        train_exp.classes_in_this_experience
        
        for (data, target, _) in train_loader:        
            data, target = data.cuda(), target.cuda()
            
            with torch.no_grad():
                output = model.module.features(data)
                
                dict['traindata'] =  torch.cat((dict['traindata'], output))
                dict['trainlabel'] =  torch.cat((dict['trainlabel'], target))

                dict_all_dataset['traindata'] =  torch.cat((dict_all_dataset['traindata'], output))
                dict_all_dataset['trainlabel'] =  torch.cat((dict_all_dataset['trainlabel'], target))


        train_set = torch.utils.data.TensorDataset(dict['traindata'], dict['trainlabel'])
        train_features.append(torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed)))
             

        for (data, target, _) in test_loader:          
            if target in train_exp.classes_in_this_experience:
                data, target = data.cuda(), target.cuda()
                
                with torch.no_grad():
                    output = model.module.features(data)

                    dict['testdata'] =  torch.cat((dict['testdata'], output))
                    dict['testlabel'] =  torch.cat((dict['testlabel'], target))

                    dict_all_dataset['testdata'] =  torch.cat((dict_all_dataset['testdata'], output))
                    dict_all_dataset['testlabel'] =  torch.cat((dict_all_dataset['testlabel'], target))

        test_set = torch.utils.data.TensorDataset(dict['testdata'], dict['testlabel'])
        test_features.append(torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=0))

    #torch.save(dict_all_dataset, featuresPath)
    return train_features, test_features


def splitFeatures(trainset, testset, n_class, start, step):
    print('Spliting features.......')
    train_features=[]
    test_features=[]

    train_features.append(torch.utils.data.DataLoader(SubDataset(trainset,[j for j in range(start)]), batch_size=bs, shuffle=True, num_workers=0))
    test_features.append(torch.utils.data.DataLoader(SubDataset(testset,[j for j in range(start)]), batch_size=bs, shuffle=False, num_workers=0))
    
    for i in range(start, n_class, step):
        a = SubDataset(trainset,[i+j for j in range(step)])
        x = torch.utils.data.DataLoader(a, batch_size=bs, shuffle=True, num_workers=0)
        train_features.append(x)
        test_features.append(torch.utils.data.DataLoader(SubDataset(testset,[i+j for j in range(step)]), batch_size=bs, shuffle=False, num_workers=0))

    return train_features, test_features

from torch.utils.data import Dataset
class SubDataset(Dataset):
    def __init__(self, original_dataset, sub_labels, target_transform=None,transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform
        self.transform=transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.transform:
            sample=self.transform(sample)
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample



if __name__ == '__main__':
    dataset = CORE50REDUCEDRESNET18(experiences = 9)

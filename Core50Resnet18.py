
from avalanche.benchmarks.classic import CORe50


from torchvision.transforms import (
    ToTensor,
    Resize,
    Normalize,
    Compose,
    RandomHorizontalFlip,
)
import os
import torch
import torchvision.models as models

from torchsummary import summary

featuresPath = './Features/Core50_resnet18.pt'
bs = 50


normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = Compose([ToTensor(), Resize(224), RandomHorizontalFlip(), normalize])
test_transform = Compose([ToTensor(), Resize(224), normalize])


def initFeaturesExtractor():

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).cuda()

    model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.Sequential(*list(model.module.children())[:-1])

    #summary(model, (3, 32, 32))
    return model

class CORE50RESNET18():
    def __init__(self, experiences = 9):
        
        self.lamda = 0.9
        self.beta = 1.1

        self.n_class = 50
        self.n_features = 512
                
        self.train_features, self.test_features = createFeatures(experiences)

def createFeatures(experiences = 9):

    model = initFeaturesExtractor()
    print('Creating features....')
    benchmark = CORe50(scenario="nc", run=experiences, mini=False, train_transform = train_transform, eval_transform = test_transform)
    
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
                output = model(data)
                output = output.view(output.size(0),-1)

                dict['traindata'] =  torch.cat((dict['traindata'], output))
                dict['trainlabel'] =  torch.cat((dict['trainlabel'], target))

                dict_all_dataset['traindata'] =  torch.cat((dict_all_dataset['traindata'], output))
                dict_all_dataset['trainlabel'] =  torch.cat((dict_all_dataset['trainlabel'], target))


        train_set = torch.utils.data.TensorDataset(dict['traindata'], dict['trainlabel'])
        train_features.append(torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0))
             

        for (data, target, _) in test_loader:          
            if target in train_exp.classes_in_this_experience:
                data, target = data.cuda(), target.cuda()
                
                with torch.no_grad():
                    output = model(data)
                    output = output.view(output.size(0),-1)

                    dict['testdata'] =  torch.cat((dict['testdata'], output))
                    dict['testlabel'] =  torch.cat((dict['testlabel'], target))

                    dict_all_dataset['testdata'] =  torch.cat((dict_all_dataset['testdata'], output))
                    dict_all_dataset['testlabel'] =  torch.cat((dict_all_dataset['testlabel'], target))

        test_set = torch.utils.data.TensorDataset(dict['testdata'], dict['testlabel'])
        test_features.append(torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=0))

    #torch.save(dict_all_dataset, featuresPath)
    return train_features, test_features
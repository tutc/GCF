from avalanche.benchmarks.classic import SplitCUB200
from torchvision import transforms
import os
import torch
import torchvision.models as models


featuresPath = './Features/cub200_resnet50'


bs = 50

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


def initFeaturesExtractor():

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).cuda()
    model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.Sequential(*list(model.module.children())[:-1])

    return model

class CUB200RESNET50():
    def __init__(self, start = 100, step = 2):
        
        self.lamda = 0.7
        self.beta = 1.4

        self.n_class = 200
        self.n_features = 2048
        global featuresPath 
        featuresPath = featuresPath + 'Step' + str(step) + '.pt'

        self.train_features, self.test_features = createFeatures(start, step)

def createFeatures(start = 100, step = 2):
    print('Creating features....')
    
    benchmark = SplitCUB200(n_experiences = ( (200 - start) // step) + 1, classes_first_batch=start, train_transform=train_transform, eval_transform=test_transform)
    
    featuresExtrator = initFeaturesExtractor()

    train_features = []
    test_features = []


    for (train_exp, test_exp) in zip(benchmark.train_stream, benchmark.test_stream):
        current_train_set = train_exp.dataset
        current_test_set = test_exp.dataset
        
        current_train_features, current_test_features = getFeatures(featuresExtrator, current_train_set, current_test_set)

        train_features.append(torch.utils.data.DataLoader(current_train_features, batch_size=bs, shuffle=True, num_workers=0))
        test_features.append(torch.utils.data.DataLoader(current_test_features, batch_size=bs, shuffle=False, num_workers=0))

    return train_features, test_features


def getFeatures(model, trainset, testset):

    mini_bs = 1
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=mini_bs, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=mini_bs, num_workers=0)


    empty = torch.tensor([]).cuda()
    dict = {'traindata': empty, 'trainlabel':empty, 'testdata': empty, 'testlabel':empty}

    model.eval()
    for (data, target, _) in train_loader:         
    
        data, target = data.cuda(), target.cuda()
        
        with torch.no_grad():
            output = model(data)
            output = output.view(output.size(0),-1)

            dict['traindata'] =  torch.cat((dict['traindata'], output))
            dict['trainlabel'] =  torch.cat((dict['trainlabel'], target))

    for (data, target, _) in test_loader:     
    
        data, target = data.cuda(), target.cuda()
        
        with torch.no_grad():
            output = model(data)
            output = output.view(output.size(0),-1)

            dict['testdata'] =  torch.cat((dict['testdata'], output))
            dict['testlabel'] =  torch.cat((dict['testlabel'], target))
    
    #saveFeatures(dict, featuresPath)

    train_set = torch.utils.data.TensorDataset(dict['traindata'], dict['trainlabel'])
    test_set = torch.utils.data.TensorDataset(dict['testdata'], dict['testlabel'])

    return train_set, test_set

def saveFeatures(dict, featuresPath = featuresPath):
    if os.path.isfile(featuresPath):
        old_dict = torch.load(featuresPath)
        old_dict['traindata'] = torch.cat((old_dict['traindata'], dict['traindata']))
        old_dict['trainlabel'] = torch.cat((old_dict['trainlabel'], dict['trainlabel']))
        old_dict['testdata'] = torch.cat((old_dict['testdata'], dict['testdata']))
        old_dict['testlabel'] = torch.cat((old_dict['testlabel'], dict['testlabel']))
        dict = old_dict
    torch.save(dict, featuresPath)
    print('features saved')
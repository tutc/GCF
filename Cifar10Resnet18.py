from avalanche.benchmarks.classic import SplitCIFAR10
from torchvision import transforms
import os
import torch
import torchvision.models as models

#from torchsummary import summary

featuresPath = './Features/cifar10_resnet18.pt'
pretrainedPath = './PretrainedModel/resnet18_ImageNet32.pth.tar'

bs = 50

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def initFeaturesExtractor():
    assert os.path.isdir('PretrainedModel'), 'Error: no pretrained directory found!'
    assert os.path.isfile(pretrainedPath) , 'Error: no pretrained file found!'
    
    checkpoint = torch.load(pretrainedPath)


    model = models.resnet18().cuda()
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model = torch.nn.DataParallel(model).cuda()
        
    model.load_state_dict(checkpoint['state_dict'])    
    model = torch.nn.Sequential(*list(model.module.children())[:-1])
    
    #summary(model, (3, 32, 32))

    return model


class CIFAR10RESNET18():
    def __init__(self, start = 2, step = 2):
        
        self.lamda = 0.8
        self.beta = 1.1

        self.n_class = 10
        self.n_features = 512
	
        experiences = self.n_class // step
        self.train_features, self.test_features = createFeatures(experiences)

def createFeatures(experiences):
    print('Creating features....')
    benchmark = SplitCIFAR10(n_experiences=experiences, train_transform=train_transform, eval_transform=test_transform)

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
    
    #torch.save(dict, featuresPath)

    train_set = torch.utils.data.TensorDataset(dict['traindata'], dict['trainlabel'])
    test_set = torch.utils.data.TensorDataset(dict['testdata'], dict['testlabel'])

    return train_set, test_set
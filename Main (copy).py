"""

Code adapted from https://github.com/Julien-pour/Dynamic-Sparse-Distributed-Memory

"""
import torch

from Clustering.KMeansClustering import kmeans

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


import numpy as np
import random

import Benchmarks as benchmarks

device = torch.device("cuda")

def E_distances(X,Y):

    return torch.sqrt( torch.sum(torch.pow(X, 2),dim=1).view(-1,1) -2 * torch.mm(X,Y.T) + torch.sum(torch.pow(Y, 2),dim=1) )



class Main(nn.Module):
    def __init__(self,Time_period, n_mini_batch,n_class=10,n_feat=160):
        super(Main, self).__init__()
        self.n_feat=n_feat
        self.n_class = n_class
        self.Time_period = Time_period
        self.ema = 2/(Time_period+1)
        self.n_mini_batch = n_mini_batch
        self.count=0
        self.beta=1.1
        
        self.Address=torch.zeros(1,n_feat).to(device)

        self.M=torch.zeros(1,self.n_class)
        self.p_norm="fro"
        
        self.num_clusters = 100
        
     
        
        
        self.global_error=0
        self.Time_period_Temperature = self.ema
        self.ema_Temperature=(2/(self.Time_period_Temperature+1))
        self.memory_global_error=torch.zeros(1)
        self.memory_min_distance=torch.zeros(1)
        self.memory_count_address=torch.zeros(1)
    
        self.acc_after_each_task=[]
        self.acc_aft_all_task=[]
        self.stock_feat=torch.tensor([]).to(device)
        self.forgetting=[]

        self.cum_acc_activ=False

        self.batch_test=True


        self.reset()

        self.check = False

    def reset(self):
        self.ema = 2/(self.Time_period+1)
        self.ema_Temperature=(2/(self.Time_period_Temperature+1))
        self.count=0
        self.Address=torch.zeros(1,self.n_feat).to(device)
        self.M=torch.zeros(1,self.n_class).to(device)
        
        
        self.global_error=0
        self.memory_global_error=torch.zeros(1)
        self.memory_min_distance=torch.zeros(1)
        self.memory_count_address=torch.zeros(1)


        self.check = False

    
    def forward(self,inputs):

        with torch.no_grad():
            out = inputs
            pred=torch.tensor([]).to(device)
            if self.batch_test:
                distance=E_distances(inputs,self.Address)
                soft_norm = F.softmin(distance/self.beta,dim=-1)
                pred = torch.matmul(soft_norm, self.M)

            else:
                for idx_x in range(len(out)):
                    x=out[idx_x]
                    distance=x-self.Address
                    norm=torch.norm(distance, p=self.p_norm, dim=-1)
                    # softmin vs argmin
                    soft_norm = F.softmin(norm/self.beta,dim=-1)
                    soft_pred = torch.matmul(soft_norm, self.M.to(device)).view(-1)
                    pred=torch.cat((pred,soft_pred.view(1,-1)),0)

        return pred
    
    
    def test_idx(self,test_dataset_10_way_split,idx_test):
        with torch.no_grad():
            total=0
            correct=0
            for idx in idx_test:
                curr_correct=0
                curr_total=0
                for batch_idx, (inputs, targets) in enumerate(test_dataset_10_way_split[idx]):
                    inputs=inputs.to(device)
                    targets=targets.type(torch.LongTensor).to(device)
                    outputs = self.forward(inputs)
                    _, predicted = torch.max(outputs,1)
                    total += targets.size(0)
                    corr=(predicted == targets).sum().item()
                    curr_correct +=corr
                    correct += corr
                    curr_total+=targets.size(0)
            accuracy=correct/total*100

        return accuracy,curr_correct/curr_total*100
    
        
    def clustering(self, label):

        _, Labels = torch.max(self.M,1)
        A = self.Address[Labels==label]
        if len(A)>=self.threshold:
            self.Address = self.Address[Labels!=label]
            self.M = self.M[Labels!=label]
        
            one_hot=torch.zeros(1,self.n_class).to(device).float()
            one_hot[0][label] = 1
            #cluster_centers = kmeans(X=A, num_clusters=self.num_clusters, distance='euclidean', device=device)
            cluster_centers = kmeans(X=A, num_clusters=self.threshold, distance='euclidean', device=device)
            self.Address=torch.cat((self.Address, cluster_centers))
            self.M=torch.cat((self.M,one_hot.repeat(len(cluster_centers),1)))
   
    def train__test_n_way_split(self, train_dataset_10_way_split, test_dataset_10_way_split,coef_global_error=1,ema_global_error=None, save_feat=False):

        count = 0
        acc_test=torch.zeros(len(train_dataset_10_way_split))
        acc_test_after_each_task_softmin = torch.zeros(len(train_dataset_10_way_split))
        self.memory_min_distance=torch.zeros(1)
        idx_seen=[]
        self.cum_acc=[]
        with torch.no_grad():
            for idx_loader in range(len(train_dataset_10_way_split)):
                idx_seen.append(idx_loader)
                for batch_idx, (inputs, targets) in enumerate(train_dataset_10_way_split[idx_loader]):
                    
                    inputs=inputs.to(device)
                    targets=targets.type(torch.LongTensor).to(device)

                    if batch_idx==self.n_mini_batch:
                        break
                    if save_feat:
                        self.stock_feat=torch.cat((self.stock_feat,inputs))
                    len_inputs = len(inputs)
                    for idx_x in range(len_inputs):
                    
                        out=inputs.to(device)
                        x=out[idx_x]
                        self.count+=1
                        distance=x-self.Address
                        norm=torch.norm(distance, p=self.p_norm, dim=-1)
                        soft_norm=norm

                        max_value, idx_max=torch.min(norm,0)

                        self.global_error+= self.ema_Temperature*(max_value-self.global_error)

                        if abs(norm[idx_max])>=self.global_error*coef_global_error:
                            
                            self.Address=torch.cat((self.Address,x.view(1,-1)))
                            targets_one_hot=F.one_hot(targets[idx_x], num_classes=self.n_class).float()

                            self.M=torch.cat((self.M,targets_one_hot.view(1,-1)))
                            self.clustering(targets[idx_x])
                        else:
                            delta_address = distance
                            soft_norm = F.softmin(norm/self.beta,dim=-1)

                            self.Address = self.Address + self.ema * torch.mul(soft_norm.view(-1,1),delta_address)
                            targets_one_hot=F.one_hot(targets[idx_x], num_classes=self.n_class).float()

                            self.M += self.ema * torch.mul(soft_norm.view(-1,1),(targets_one_hot-self.M))   
                    

                if self.cum_acc_activ:
                    acc,last_acc = self.test_idx(test_dataset_10_way_split,idx_seen)
                    self.cum_acc.append(acc)
                    acc_test_after_each_task_softmin[idx_loader] = last_acc


            self.acc_after_each_task=acc_test_after_each_task_softmin
            acc_test_after_each_task_softmin=acc_test_after_each_task_softmin

            acc_test_after_all_task_softmin=torch.zeros(len(train_dataset_10_way_split))
            acc_test=torch.zeros(len(train_dataset_10_way_split))

            # test after learn all task

            for idx_loader in range(len(test_dataset_10_way_split)):
                count_input=0
                correct=0
                correct_softmin=0
                if self.batch_test:
                    acc,last_acc = self.test_idx(test_dataset_10_way_split,[idx_loader])
                    acc_test[idx_loader]=0 
                    acc_test_after_all_task_softmin[idx_loader]=last_acc
                else:
                    for batch_idx, (inputs, targets) in enumerate(test_dataset_10_way_split[idx_loader]):
                        out=inputs.to(device)
                        targets=targets.type(torch.LongTensor)
                        for i in range(len(out)):
                            count_input += 1
                            x=out[i]
                            distance=x-self.Address
                            norm=torch.norm(distance, p=self.p_norm, dim=-1)
                            #softmin
                            soft_norm=F.softmin(norm/self.beta,dim=-1)
                            soft_pred = torch.matmul(soft_norm, self.M.to(device)).view(-1)

                            arg_soft_pred = torch.argmax(soft_pred)

                            #argmin
                            idx=torch.argmin(norm)

                            pred = torch.argmax(self.M[idx])
                            if pred==targets[i]:
                                correct += 1
                            if arg_soft_pred==targets[i]:
                                correct_softmin += 1
                    acc_test[idx_loader]=correct/count_input*100
                    acc_test_after_all_task_softmin[idx_loader]=correct_softmin/count_input*100



                self.acc_aft_all_task=acc_test_after_all_task_softmin

            return acc_test_after_each_task_softmin, acc_test_after_all_task_softmin
        
    
    def grid_search_spread_factor(self, Time_period, n_mini_batch, train_dataset_10_way_split, test_dataset_10_way_split,N_try=5,ema_global_error="same",coef_global_error=1,random_ordering=True):

        acc_test=torch.zeros(N_try,len(train_dataset_10_way_split))


        cum_acc=torch.zeros(N_try,len(train_dataset_10_way_split))

        acc_test_softmin=torch.zeros(N_try,len(train_dataset_10_way_split))
        accuracy_mean_test_softmin=torch.zeros(len(train_dataset_10_way_split))
        accuracy_std_test_softmin=torch.zeros(len(train_dataset_10_way_split))
        N_address_use=torch.zeros(N_try)
        self.forgetting=[]

        self.n_mini_batch=n_mini_batch
        self.Time_period = Time_period
        self.ema = 2/(Time_period+1)

        
        for idx_try in tqdm(range(N_try)):
            self.reset()
            Acc_test, Acc_test_softmin = self.train__test_n_way_split(train_dataset_10_way_split, test_dataset_10_way_split,ema_global_error=ema_global_error,coef_global_error=coef_global_error)
            if self.cum_acc_activ:
                cum_acc[idx_try]=torch.tensor(self.cum_acc)
            self.forgetting.append((self.acc_after_each_task*100-self.acc_aft_all_task).mean())
            acc_test[idx_try]=Acc_test
            acc_test_softmin[idx_try]=Acc_test_softmin
            N_address_use[idx_try] = self.M.size(0)
            
            
            #random ordering
            if random_ordering:
                dataset_shuffle = list(zip(train_dataset_10_way_split, test_dataset_10_way_split))
                random.shuffle(dataset_shuffle)
                train_dataset_10_way_split, test_dataset_10_way_split = zip(*dataset_shuffle)

        accuracy_mean_test_softmin = acc_test_softmin.mean(0)
        acc_soft_mean=acc_test_softmin.mean(1)

        accuracy_std_test_softmin = acc_test_softmin.std(0)
#Tu        print("forgetting softmin inference = {:.1f} % ± {:.1f}".format(np.mean(self.forgetting),np.std(self.forgetting)))

        return acc_soft_mean,N_address_use.mean(),acc_test,cum_acc


def run(data, memorysize = 3000):    

    bs = 50
    N_try = 5
    n_mini_batch = 55

    Time_period = 500
    Time_period_temperature = 150
    exp = Main(Time_period ,n_mini_batch,n_class=data.n_class,n_feat=data.n_features) 
    exp.p_norm = "fro"
    exp.beta = data.beta 
    exp.cum_acc_activ = True
    exp.Time_period_Temperature = Time_period_temperature
    random_ordering = True
    exp.threshold = (int)(memorysize /data.n_class)
    #print("threshold: ")
    #print(exp.threshold)
    accuracy_c100_100w,N_address_use_c100_100w,acc_test_softmin,cum_sum = exp.grid_search_spread_factor(Time_period, n_mini_batch, data.train_features,data.test_features,N_try,ema_global_error="diff",coef_global_error=data.lamda, random_ordering = random_ordering)
    
    last_acc = float(accuracy_c100_100w.mean())    
    avg_acc = float(cum_sum.mean())
    mem_size = int(len(exp.Address))



    return avg_acc, last_acc, mem_size

if __name__ == '__main__':

    #dataset = benchmarks.Cifar10ReducedResnet18.CIFAR10REDUCEDRESNET18(start = 2, step = 2) #Done
    #dataset = benchmarks.Cifar10Resnet18.CIFAR10RESNET18(start = 2, step = 2) #Done
    #dataset = benchmarks.Core50ReducedResnet18.CORE50REDUCEDRESNET18() #Done
    dataset = benchmarks.Core50Resnet18.CORE50RESNET18() #Done
    avg_acc, last_acc, mem_size = run(dataset, 10000)

    #dataset = benchmarks.Cub200Resnet50.CUB200RESNET50(start = 2, step = 2) #Done
    #dataset = benchmarks.Cub200Resnet50.CUB200RESNET50(start = 5, step = 5) #Done
    #dataset = benchmarks.Cifar100Resnet50.CIFAR100RESNET50(start = 2, step = 2) #Done
    #dataset = benchmarks.Cifar100Resnet50.CIFAR100RESNET50(start = 5, step = 5) #Done
    #avg_acc, last_acc, mem_size = run(dataset, 3000)

    print('Average accuracy: ', avg_acc)
    print('Last accuracy: ', last_acc)
    print('Memory used: ', mem_size)

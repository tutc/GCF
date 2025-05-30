import argparse

import Main as experiment
import Benchmarks as benchmarks

def main(args):
    print(args)

    if args.dataset == 'cifar10':
        if args.backbone == 'reduced':
            dataset = benchmarks.Cifar10ReducedResnet18.CIFAR10REDUCEDRESNET18()
        else:
            dataset = benchmarks.Cifar10Resnet18.CIFAR10RESNET18()
    elif args.dataset == 'core50':
        if args.backbone == 'reduced':
            dataset = benchmarks.Core50ReducedResnet18.CORE50REDUCEDRESNET18()
        else:
            dataset = benchmarks.Core50Resnet18.CORE50RESNET18()
    elif args.dataset == 'cifar100':
        dataset = benchmarks.Cifar100Resnet50.CIFAR100RESNET50(start = args.step, step = args.step)
    else:
        dataset = benchmarks.Cub200Resnet50.CUB200RESNET50(start = args.step, step = args.step)



    avg, last, meansize = experiment.run(dataset, args.memory_size)
    print('Avg: ',avg)
    print('Last: ',last)
    print('Memory size: ',meansize)

if __name__ == "__main__":

    # Commandline arguments
    parser = argparse.ArgumentParser(description="GCF....")
    
    #parser.add_argument('--num_runs', dest='num_runs', default=1, type=int, help='Number of runs (default: %(default)s)')
    parser.add_argument('--dataset',  dest='dataset', default='cifar100', type=str, help='Dataset')
    parser.add_argument('--backbone',  dest='backbone', default='resnet50', type=str, help='Features Extractor')
    parser.add_argument('--step',  dest='step', default=2, type=int, help='Step size')
    parser.add_argument('--memory',  dest='memory_size', default=3000, type=int, help='Memory size')

    args = parser.parse_args()
    main(args)

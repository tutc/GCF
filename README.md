# Dynamic content-addressable memory based on global centroid features for online task-free continual learning
## Dataset
- Split CIFAR-10
- Split CIFAR-100
- Split CUB-200
- CORe-50
## Feature extractor
- Reduced Resnet-18
- Resnet-18
- Resnet-50
## Sample commands to run GCF
##### Dataset: Split CIFAR-10, Feature extractor: Reduced Resnet-18, Memory size: 1000
<pre>
  <code id="code-snippet">
    python General_main.py --dataset cifar10 --backbone reduced --memory 1000
  </code>
</pre>
##### Dataset: CORe-50, Feature extractor: Resnet-18, Memory size: 2000
<pre>
  <code id="code-snippet">
    python General_main.py --dataset core50 --backbone resnet18 --memory 2000
  </code>
</pre>
##### Dataset: Split CIFAR-100, Feature extractor: Resnet-50, Step: 2
<pre>
  <code id="code-snippet">
    python General_main.py --dataset cifar100 --backbone resnet50 --step 2
  </code>
</pre>
##### Dataset: Split CUB-200, Feature extractor: Resnet-50, Step: 5
<pre>
  <code id="code-snippet">
    python General_main.py --dataset cub200 --backbone resnet50 --step 5
  </code>
</pre>

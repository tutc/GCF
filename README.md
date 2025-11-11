# Dynamic content-addressable memory based on global centroid features for online task-free continual learning

## Abstract
In online continual learning, a neural network learns from a continuous data stream in which each data point is processed only once and never revisited in the next learning stages. This non-revisiting mechanism often leads to Catastrophic Interference (CI), i.e., the model can dramatically lose the learned features of previous tasks when new information is learned. This forgetting issue becomes even more severe in online task-free continual learning, where the model has to adapt to the latest information autonomously without knowing task boundaries in advance. Furthermore, the inner sparse distributed memory used to store data points can become overburdened due to the rapid growth of new data points that need to be stored for upcoming tasks.
To deal with these challenges, in this article, we propose an efficient model for online task-free continual learning built on the following two novel concepts. Firstly, we introduce a new powerful operator based on Global Centroid Features (GCF) to take advantage of the condensed knowledge of data points stored in dynamic content-addressable memory. Thanks to a suitable clustering mechanism, the GCF accumulation is able to control the storage overload issue caused by the rapid increase of newly stored data points in the upcoming tasks. Secondly, we design a continual learning model that effectively leverages the extraction of GCF-based information, enabling it to alleviate catastrophic forgetting through cognitive condensation of GCF. Thereby, the proposed GCF-based model simultaneously exploits the global centroid information of the stored data points and efficiently manages the rapid growth of stored data points in the storage space. The experimental results on benchmark image classification datasets emphasize the superior performance of our approach compared to state-of-the-art methods.

**Keywords:** Online task-free continual learning, Class-incremental learning, Clustered features, Image classification
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

## Citation
If you use this code in your research, please cite the following relevant work:
<pre>
  <code id="code-snippet">
    @article{tutc_GCF,
      author       = {Cong Tu Tran, Thanh Tuan Nguyen, Thanh Phuong Nguyen, and Nadège Thirion-Moreau},
      title        = {Dynamic content-addressable memory based on global centroid features for online task-free continual learning},
      journal      = {Machine Vision and Applications},
      note         = {Submitted 2025}
    }  </code>
</pre>
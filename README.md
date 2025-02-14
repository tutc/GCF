# Dynamic content-addressable memory based on global centroid features for online task-free continual learning

## Abstract
In online continual learning, a neural network can learn from a never-ending stream of data, where each data point would be processed once and not be revisited in the next learning stages. Due to the non-revising mechanism, the model
can dramatically lose the learned features of previous tasks when it learns new information. This catastrophic interference would become more serious in online task-free continual learning, where the model has to adapt to the new information without knowing task boundaries in advance. Furthermore, the inner sparse
distributed memory for stored data points would be overburdened due to the sharp growth of new data points needed to be stored for upcoming tasks. To deal
with these problems, an efficient model for online task-free continual learning is introduced subject to two following novel concepts. Firstly, a powerful operator of
global centroid features (GCFs) is proposed to take advantage of the condensed knowledge of data points stored in dynamic content-addressable memory. Thanks
to a suitable clustering mechanism, the GCF accumulation is able to control the overloaded storage issue caused by the accelerated increase of new stored data
points in the upcoming tasks. Secondly, an effective continual learning model is introduced to take into account the extraction of GCF-based information. It
can emphatically deal with the catastrophic forgetting challenge thanks to the cognitive condensation of GCFs. Thereby, the proposed GCF-based model can take advantage of the global centroid information of the stored data points, and efficiently manage the sharp growth of stored data points in the storage space, concurrently. Experimental results for image classification on benchmark datasets have proved the prominent efficacy of our proposals in comparison with exist- ing methods. 
**Keywords:*** Online task-free continual learning, Class-incremental learning, Clustered features, Image classification
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
      journal      = {Machine Learning},
      note         = {Submitted 2025}
    }  </code>
</pre>


from avalanche.benchmarks.classic import SplitMNIST, CORe50, SplitCUB200

"""
bm1 = SplitMNIST(
    n_experiences=5,  # 5 incremental experiences
    return_task_id=True,  # add task labels
    seed=1  # you can set the seed for reproducibility. This will fix the order of classes
)

bm = CORe50(
    scenario = 'nc',
    run = 9,  # 5 incremental experiences
    #return_task_id=True,  # add task labels
    #seed=1  # you can set the seed for reproducibility. This will fix the order of classes
)
bm = SplitCUB200(n_experiences=21, classes_first_batch=100, seed = 1234)
# streams have a name, used for logging purposes
# each metric will be logged with the stream name
print(f'--- Stream: {bm.train_stream.name}')
# each stream is an iterator of experiences
for t, exp1 in enumerate(bm.train_stream):
    # experiences have an ID that denotes its position in the stream
    # this is used only for logging (don't rely on it for training!)
    eid = exp1.current_experience
    # for classification benchmarks, experiences have a list of classes in this experience
    clss = exp1.classes_in_this_experience
    # you may also have task labels
    tls = exp1.task_labels
    #print(f"EID={eid}, classes={clss}, tasks={tls}")
    # the experience provides a dataset
    print(f"data: {len(exp1.dataset)} samples")
    #for exp in bm.test_stream:
    exp2 = bm.test_stream[0]
  
    class_exp = exp2.classes_in_this_experience[clss[0]:clss[-1]+1]

    print(f"EID={exp2.current_experience}, classes={class_exp}, task={tls}")
    print(f"test: {len(exp2.dataset)} samples")

#for exp in bm.test_stream:
    
#    print(f"EID={exp.current_experience}, classes={exp.classes_in_this_experience}, task={tls}")


"""

from avalanche.benchmarks.classic import CORe50, SplitTinyImageNet, SplitCIFAR10, \
    SplitCIFAR100, SplitCIFAR110, SplitMNIST, RotatedMNIST, PermutedMNIST, SplitCUB200

# creating the benchmark (scenario object)
perm_mnist = PermutedMNIST(
    n_experiences=3,
    seed=1234,
)

# recovering the train and test streams
#train_stream = perm_mnist.train_stream
#test_stream = perm_mnist.test_stream

ds = SplitCUB200(n_experiences=40, classes_first_batch=5, seed = 1234)

#ds = CORe50(scenario = 'nc', run=9)
train_stream = ds.train_stream
test_stream = ds.test_stream
i = 0
# iterating over the train stream

for (experience, test_exp) in zip(ds.train_stream, ds.test_stream):

#for experience in train_stream:
    print("Start of task ", experience.task_label)
    print('Classes in this task:', experience.classes_in_this_experience)

    # The current Pytorch training set can be easily recovered through the 
    # experience
    current_training_set = experience.dataset
    # ...as well as the task_label
    print('Task {}'.format(experience.task_label))
    print('This task contains', len(current_training_set), 'training examples')
    #if i==0:
    # we can recover the corresponding test experience in the test stream
    #current_test_set = test_stream[experience.current_experience].dataset
    current_test_set = test_exp.dataset
    print('This task contains', len(current_test_set), 'test examples')
    print('Classes in this test task:', test_exp.classes_in_this_experience)


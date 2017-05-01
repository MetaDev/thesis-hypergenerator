# Hyper-generator
## Search space optimisation for procedural content generators

The hyper-generator is a prototype for the generative system proposed in my master thesis (master computer science, 2016 University of Ghent). 
In the system the user has to define the objects to be generated in two ways.
First, the objects search space is defined as a hiearchical model of its parameters (prior knowledge) and secondly with objective functions on these parameters (posterior knowledge).
If the user defines a relation between parameters as probabilistic (binary choice), than the system will learn the specific conditional probability between parameters based on their respective objective functions.
These conditional relations are modelled with Gaussian mixture models and fitted with the EM algorithm.

The [thesis book](https://github.com/MetaDev/thesis-hypergenerator/blob/master/doc/HaraldDeBondt_2016_Thesis.pdf) contains all relevant information surrounding my thesis (100 pages), the [thesis extended abstract](https://github.com/MetaDev/thesis-hypergenerator/blob/master/doc/extended_abstract_HDB.pdf) is a 5 page summary in "scientific paper" format.

## Installation

Run *pip install -r requirements.txt* in your shell from the root folder

Tested on OSX with Python 3.5.1 |Anaconda 2.5.0 (x86_64)

## Usage

All experiments performed in the thesis are located in the file *learning/training_test.py*, it contains the code to save the results to files.
Running this script will perform a set of training operations with pre-configured models and training parameters.
These are respectively located in *model/test_models.py* and *training/train_gmm.py*
The result is a collection of figures visualising the trained probabilistic models and relevant run-time and training information.
The figures and information are respectively saved in *pdf* and *txt* files.

An example configuration method and sample output (I refer to the thesis book for an elaborate explenation of experiments and results):

```python
def test_n_component():
    training_params.title="test for number of components in gmm "
    print(training_params.title)
    print()
    for n_components in np.arange(10,50,10):
        training_params.n_components=n_components
        training_params.extra_info="GMM number of components: " + str(n_components)
        print("GMM number of components: ",n_components)
        time_score_all_models()

    print_train_parameters()
    training_params.reset()
```
Snippet from *test11.txt*:

```
test for number of components in gmm 

GMM number of components:  10
model 0
score before training:  0.620076270819
fitness before training iteration  0
seperate fitness
Target distances
total mean= 0.605315: variance= 0.017267
iteration  0  trial score mean:  0.666903202662  variance:  0.000369106788162
improved selected with score:  0.693047215309
fitness parameters,
func: <function min_dist_sb at 0x18a859400>
threshhold: 0
order: 4
fitness_relation: Fitness_Relation.pairwise_siblings
target: 2
name: Target distances
regression_target: 1

,

model parameters
parent variables, ['shape0']
sibling variables, ['position']

model 1
score before training:  0.414077565011
fitness before training iteration  0
seperate fitness
polygon overlay
total mean= 0.407367: variance= 0.048244
iteration  0  trial score mean:  0.73160785765  variance:  0.0141653088633
improved selected with score:  0.817439960052
fitness parameters,
func: <function norm_overlap_pc at 0x18a84b2f0>
threshhold: 0
order: 32
fitness_relation: Fitness_Relation.pairwise_parent_child
name: polygon overlay
regression_target: 1

,
```

Snapchots from *test11.pdf*:

![](https://github.com/MetaDev/thesis-hypergenerator/blob/master/doc/example_output_0.png)
![](https://github.com/MetaDev/thesis-hypergenerator/blob/master/doc/example_output_2.png)

# App Recommendation System

This project demonstrates a recommendation system for applications based solely
on their app store properties. According to the specifications, 
no labeled data, such as which apps were installed by the same user, 
or which previous recommendations were accepted. Therefore, we cannot use
methods such as Collaborative Filtering, or classifier-based recommendations.
Instead, we describe below a method based on document similarity, where the
training corpus consists of app descriptions.

The rest of this README includes the following sections: First, how to install
the program. Second, the stages of the method and how to run each of them.
Third, the recommendation method itself and the rationale behind it.
Finally, the alternatives and improvements that were considered, but not
used or included in the project.

## Installation

The included code requires some dependencies, that can be easily
installed using the standard Python repository:
```
pip install configargparse numpy pandas pytorch scipy sklearn tqdm
```

For optimal performance during the training stage (see below), 
*pytorch* should be installed with CUDA and ran on a GPU machine.
However, for reasonably-sized data sets, a CPU alone would do. 
Particularly, it is not required to run *pytorch* on a GPU after the code
is deployed as a service.

## Stages

Executing this project consists of the following stages:
1. Collecting data for training
1. Training the model
1. Using the trained model in the recommendation function

### Collecting data
The required data for this project would be a large collection of app
properties, in the same format as returned by the app store lookup API call.
One way to obtain this data, suggested in 
[a blog post by Colin Eberhardt](https://blog.scottlogic.com/2014/03/20/app-store-analysis.html), 
is to harvest the search API in brute force, running over a series of
three-letter prefixes. Some sample shell commands are listed below.
Apple app store enforces an unknown throttle limit. This code was proven
to work, but is somewhat slow, taking a few hours to complete. 
A delay smaller than 1 second, or a parallel approach, may be faster,
but may also encounter 403 errors.

```bash
for x in {a..z}{a..z}{a..z}; do 
  curl -s "https://itunes.apple.com/search?country=us&entity=software&term=${x}" > ${x}.json; 
  sleep 1; 
done
```

A better way to obtain this data would be to register to iTunes' affiliates
program, but it was out of the scope of this task.

#### Cleaning the data

The training stage expects to read a single gzipped file containing a JSON
dictionary of apps properties. However the raw data from the above script
will result in multiple small files, potentially containing duplicates.
I attached a small script called `prep.py` that takes the path expression
to these file and the output file name, and makes the needed transformation.
It is advised to gzip the result. 

### Training the model

The model training procedure is found in the `train.py` file. Its input is 
a JSON file consisting of a single dictionary mapping from app ids to their
properties, as explained in the previous section. Additionally, the training
function takes many hyper-parameters, listed with their defaults in the
`train.conf` file. The output includes two files: 
- `word_to_idx.pickle` a dictionary file mapping common words to numerical ids,
in the Python standard serialization format
- `embed.npy` a file containing the word embeddings weights matrix,
in the Numpy binary serialization format.

The exact names for these files can be set using configuration argument. 

### Recommending

Once the index and embedding files are ready, one can run the main API 
function found in the `Recommend` class in the `recommend.py` file.
When ran as main, it takes the installed app id as the first argument,
and the eligible apps ids as the consequent arguments. For example
```shell
python recommend.py 429775439 506627515 504720040 447553564 440045374 512939461
```
will execute the example from the specifications, where 429775439 is the id
of the installed app. Notice that the code folder has to be in the env.
`PYTHONPATH` for the local imports.

## Method

As mentioned in the introduction, we do not have labeled data
in this project. Therefore we can not mine the data for guidelines,
or use it to tune a model. However, since the app store exposes a search API
(albeit subject to undocumented throttle limits), we can harvest it for
*unlabeled* data, namely the properties and descriptions of other apps.

When considering the task at hand, we have no knowledge that can directly
indicate if the user is more or less likely to be interested in the
eligible apps. We can only estimate how similar is one of these apps to the
current app. There are several parameters that can contribute to this 
similarity, such as app ranking or supported devices. However, the property
which offers the richest information is the textual description.

Therefor I decided to use textual similarity between the descriptions as the
primary score for recommendation. Supposedly, a user would be most interested
in apps that share the same topics as the one he or she already installed.

The textbook method for calculating document similarity is TF-IDF, when
the IDF boost factors are taken from a typical corpus - harvested app
descriptions, in this case. However, I have decided not to use this method.
Simple term-based similarity fall short when trying to represent a *topic*.
For example, TF-IDF would identify two documents discussing cars as similar,
but not necessarily "car" and "automobile". 

Instead, the training stage performs **word2vec** encoding. It trains a 
neural network to guess the missing word in a context 
(using either n-grams or CBOW), and then saves the weights of the embeddings
layer to a file. It is analogous to saving, e.g., the IDF boost of each word,
only that instead of a single number per word, it saves dense vector.

At the actual recommendation stage, the function calculates for each app
description the normalized histogram of the embeddings of its words. 
Then the final recommendation score is the cosine similarity between
the two normalized histogram vectors.

## Roads not taken 

I have overlooked several possible alternatives and enhancements during
the implementation of this project. Firstly, I decided to use embedding
instead of classic text IR metrics. Ostensibly because embeddings handle
topics better, but also as an opportunity to learn pytorch.

I implemented the recommendation score solely on textual similarity. In
real-world settings, other properties, such as ranking, categories, etc., 
should have been factored in as well. I overlooked these in the interest 
of time, and also to emphasis the more novel approach.

Finally, the training algorithm itself could have been enhanced. 
For example, instead of a loss function based on guessing a single word,
one could consider an RNN architecture accumulating state for guessing the
category of the entire document. This alternative was also dropped due to
the limited time.
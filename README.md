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

### Training the model

### Recommending

## Method

## Roads not taken 
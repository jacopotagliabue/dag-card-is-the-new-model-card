# dag-card-is-the-new-model-card
Template-based generation of DAG cards from Metaflow classes, inspired by Google cards for machine learning models.

## Overview
WIP

## How to Run It
Code has been developed and tested on Python 3.7; dependencies are specified in the `requirements.txt` file. 
Please create a local `.env` file based on the provided template, and fill it with your values.

### Prerequisites

* Make sure [Metaflow](https://metaflow.org/)
 with [s3 support](https://docs.metaflow.org/metaflow-on-aws/metaflow-on-aws) is up and running.
* Get a valid API key from [Weights & Biases](https://wandb.ai/site).

### Sample DAG
Assuming you are using named profiles for Metaflow, you can run the DAG with:

`METAFLOW_PROFILE=my_profile python training_flow.py run`

The DAG is mostly just a simplified version of the one in our [previous tutorial](https://github.com/jacopotagliabue/no-ops-machine-learning/tree/main/serverless);
as such, it is just built for pedagogical purposes with some shortcuts here and there (e.g. re-using the local
model folder to run behavioral tests).

### Card Builder
Assuming you are using named profiles for Metaflow, you can create a DAG card with:

`METAFLOW_PROFILE=my_profile python card_builder.py`

The result will be a static HTML in the `card` folder.

## Acknowledgements

* Google cards were first presented at [FAT*](https://arxiv.org/abs/1810.03993), 
and our general styling was influenced 
by their [examples](https://modelcards.withgoogle.com/face-detection).
* Metaflow functionalities comes from their standard client plus some
 creative digging into their [repo](https://github.com/Netflix/metaflow/tree/master/metaflow).
* Charts are simple scripts embedded in the page, 
all built with out of the box functions from [Chart.js](https://www.chartjs.org/).
* Table style is from [here](https://dev.to/dcodeyt/creating-beautiful-html-tables-with-css-428l).


## Open Points / Backlog

In no particular order, some open points and improvements to make the card builder a little less hackish 
(together with the TODOs already in the code, of course).

* include a visualization of the model, even if it's just the standard Keras-generated pic.

## License
This code is provided "as is" and it is licensed under the terms of the MIT license.
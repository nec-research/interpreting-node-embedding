## Interpreting Node Embedding Source Code

This repository contains code for the paper [Interpreting Node Embedding with Text-labeled Graphs](https://ieeexplore.ieee.org/document/9533692) (Serra, Xu, Lawrence, Niepert, Tino & Yao, IJCNN 2021).

For each text-labeled graphs, the code contains all the necessary steps to reproduce the experiments. The scripts are implemented with Python 2.7, and have been tested with Linux OS.


After data preparation and subsequent vocabulary selection, our software comprises two major parts:
1. Generation of the textual explanations with our implemented model.
2. Evaluation of the results through both quantitative and qualitative experiments.


### Data sets
In the repository we uploaded a small product category for experimentation, i.e. *Patio*. The raw reviews are contained in the directory `data/patio/reviews`.
All the product categories are publicly available [here](https://jmcauley.ucsd.edu/data/amazon/). Please note that we use the 5-core version of these data sets.

### Dependencies
All the dependencies are installed if `pip install -r requirements.txt` is run. 

### Data preprocessing
Before training, some data preparation is needed to run the architecture. This includes data cleaning, vocabulary selection and data splitting. To preprocess the data, run the following bash command:
 - `bash preprocess_data.sh`

### Training
Once we prepared the data, we can run the architecture. To train the architecture, run the following command:
 - `python run_ignn.py`

The python file `utils.py` contains paths, hyperparameters and functions needed to run the steps above. The list of datasets to evaluate can be changed in this file. Please, ensure to first download the corresponding raw review data from the link provided before.

**Input**
- `users_map.pkl`: dictionary of the form `{userID: index}`
- `products_map.pkl`: dictionary of the form `{productID: index}`
- `{}_train.pkl`: replace `{}` with either `users_ID`, `products_ID`, `words`, `ratings`. Lists of training data for users, products, ratings and reviews (i.e. biterm lists).
- `{}_test.pkl`: replace `{}` with either `users_ID`, `products_ID`, `words`, `ratings`. Lists of test data for users, products, ratings and reviews (i.e. biterm lists).
- `keywords_mat.pkl`: file containing the $`V \times D`$ vocabulary matrix, i.e. the vector representations of the words contained in the vocabulary (note that the vector representations are taken from the pretrained language model).
 
**Output**
- `beta.pkl`: the file stores the $`\beta`$ matrix. This matrix will be used for generating textual explanations for the considered nodes.
- `z_users.pkl`: the file stores $`\theta_{i, k}`$, i.e. the probabilities of user $i$ to belong to cluster $k$. For all users and user clusters.
- `z_prods.pkl`: the file stores $\theta_{j, \ell}$, i.e. the probabilities of product $j$ to belong to cluster $\ell$. For all products and product clusters.
- `mse_evaluation.csv`: the file contains the train and test mean squared error (MSE) values for each evaluated epoch.
- `nll_evaluation.csv`: the file contains the train and test negative log-likelihood (NLL) values for each evaluated epoch.

### Results evaluation
We can use $`\beta`$, $`\theta_{i, k}`$ and $`\theta_{j, \ell}`$ to generate textual explanations for nodes. For more details about the generation of the explanations, we refer the readers to the paper. To visualize and evaluate the results using the output of our model, we provided a Jupyter notebook file `results_visualization.ipynb`. This file contains all the instructions to reproduce the images contained in the paper, and to manually explore the results.

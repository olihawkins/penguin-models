# penguin-models

This is a small Python project which explores using scikit-learn to classify penguins by species in the [Palmer penguins](https://github.com/allisonhorst/palmerpenguins) dataset using their bill features. This was a personal project that I used to learn about using support vector machines in scikit-learn. 

## Setup

To run the code you will need a Python 3 installation with the packages listed in `environment.yml`. To create an environment with these packages using the [Anaconda](https://docs.anaconda.com/anaconda/install/) distribution, run the following `conda` command in the repo directory:

```bash
conda env create -f environment.yml
```

This will create an environment called `penguin-models`. You can activate the environment with:

```bash
conda activate penguin-models
```

And deactivate it with:

```zsh
conda deactivate
```

See the [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for further information on environments.

## Analysis

To run the analysis, start an IPython shell:

```bash
ipython
```

Then import the `analysis` module and call its `run` method:

```python
import analysis
analysis.run()
```

This will load the data, train the models, and create the plots in the `plots` directory. There in an `index.html` file in the `plots` directory that shows all of the plots in an annotated webpage.

## Style

The plots use a custom matplotlib theme called `eda`. In the `plots` module this is loaded from the file `style/eda.mplstyle`. 

If you want to use this style in other projects, you can copy the file into your matplotlib style library, which is normally located at `~/.matplotlib/stylelib`. You can then load it with:

```python
import matplotlib.pyplot as plt
plt.style.use(['eda'])
```

## Further reading

I've been learning how to use scikit-learn with Aurelion Geron's book [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.co.uk/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646). It's _really_ good. This project was produced by applying what he teaches in this book to a novel dataset.
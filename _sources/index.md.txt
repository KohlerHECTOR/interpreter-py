# Interpreter: Interpretable and Editable Programmatic Tree Policies for Reinforcement Learning
## Imitation Learning Context
In imitation learning, the goal is to train a policy (in this case, a decision tree) that mimics the behavior of an expert policy (typically a neural network). The expert policy provides demonstrations (state-action pairs), which the imitator uses to learn how to act in the environment.

## Traditional Decision Trees
Traditional decision trees split the data using single features at a time. For instance, a split might be based on whether feature_1 > threshold. This can be limiting, as it only considers the value of one feature independently when making decisions.

## Oblique Decision Trees
Oblique decision trees, on the other hand, use linear combinations of multiple features to make splits. A decision rule in an oblique decision tree might look like a1 * feature_1 + a2 * feature_2 + ... + an * feature_n > threshold, where a1, a2, ..., an are coefficients. This allows the tree to create more complex, non-axis-aligned decision boundaries, which can capture interactions between features.

## Oblique Data Generation in Imitation Learning
To train an oblique decision tree, the feature space is often transformed to include additional features that represent linear combinations or interactions between the original features. This enriched feature space can help the tree model more complex patterns in the data, similar to those captured by a neural network.

## ObliqueDTPolicy Class
In the provided ```ObliqueDTPolicy``` class, the method get_oblique_data generates this enriched feature space by including pairwise differences between features.

## Usage


Information on how to use Interpeter library.

```{toctree}
:maxdepth: 2

usage.md
```


## API

Information on python classes behind Interpreter.

```{toctree}
:maxdepth: 2

api
```

### Cite

```bibtex
@misc{kohler2024interpretableeditableprogrammatictree,
      title={Interpretable and Editable Programmatic Tree Policies for Reinforcement Learning}, 
      author={Hector Kohler and Quentin Delfosse and Riad Akrour and Kristian Kersting and Philippe Preux},
      year={2024},
      eprint={2405.14956},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2405.14956}, 
}
```

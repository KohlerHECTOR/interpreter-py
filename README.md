# Interpretable and Editable Programmatic Tree Policies for Reinforcement Learning
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


# Usage
```bash
pip install git+https://github.com/KohlerHECTOR/interpreter-py
```

```python
from interpreter import Interpreter
from interpreter import ObliqueDTPolicy, SB3Policy, DTPolicy

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from sklearn.tree import DecisionTreeRegressor
from huggingface_sb3 import load_from_hub

from pickle import dump, load

# Download a policy from the stable-baselines3 zoo
checkpoint = load_from_hub(
	repo_id="sb3/sac-HalfCheetah-v3",
	filename="sac-HalfCheetah-v3.zip"
)

# Load the oracle policy
env = gym.make("HalfCheetah-v4")
model = SAC.load(checkpoint)
oracle = SB3Policy(model.policy)

# Get oracle performance
print(evaluate_policy(oracle, Monitor(env))[0])

# Instantiate the decision tree class (here a regression tree with at most 16 leaves)
clf = DecisionTreeRegressor(max_leaf_nodes=32) # Change to DecisionTreeClassifier for discrete Actions.
tree_policy = DTPolicy(clf, env) # 
# You can replace by ObliqueDTPolicy(clf, env) for more performing but less interpretable.

# Start the imitation learning
interpret = Interpreter(oracle, tree_policy, env)
interpret.train(10)

# Eval and save the best tree
best_tree_policy, _ = interpret.get_best_tree_policy()
final_tree_reward, _ = evaluate_policy(best_tree_policy, env=env, n_eval_episodes=10)
print(final_tree_reward)
# Here you can replace pickle with joblib or cloudpickle
with open("tree_halfcheetah.pkl", "wb") as f:
    dump(best_tree_policy.clf, f, protocol=5)

with open("tree_halfcheetah.pkl", "rb") as f:
    clf = load(f)
# Render
evaluate_policy(DTPolicy(clf, env), env=Monitor(gym.make("HalfCheetah-v4", render_mode="human")), render=True)
```

# Cite
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
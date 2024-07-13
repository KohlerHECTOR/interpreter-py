## Installation
```bash
pip install git+https://github.com/KohlerHECTOR/interpreter-py.git@v0.1.5
```


## Quickstart
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
    repo_id="sb3/sac-HalfCheetah-v3", filename="sac-HalfCheetah-v3.zip"
)

# Load the oracle policy
env = gym.make("HalfCheetah-v4")
model = SAC.load(checkpoint)
oracle = SB3Policy(model.policy)

# Get oracle performance
print(evaluate_policy(oracle, Monitor(env))[0])

# Instantiate the decision tree class (here a regression tree with at most 16 leaves)
clf = DecisionTreeRegressor(
    max_leaf_nodes=32
)  # Change to DecisionTreeClassifier for discrete Actions.
tree_policy = ObliqueDTPolicy(clf, env)  #
# You can replace by DTPolicy(clf, env) for interpretable axis-parallel DTs.

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
evaluate_policy(
    ObliqueDTPolicy(clf, env),
    env=Monitor(gym.make("HalfCheetah-v4", render_mode="human")),
    render=True,
)

```
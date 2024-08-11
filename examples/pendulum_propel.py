from interpreter import Propel
from interpreter import ObliqueDTPolicy

from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym
from sklearn.tree import DecisionTreeRegressor

from pickle import dump, load

# Load the oracle policy
env = gym.make("Pendulum-v1")
# Instantiate the decision tree class (here a regression tree with at most 16 leaves)
clf = DecisionTreeRegressor(
    max_leaf_nodes=32
)  # Change to DecisionTreeClassifier for discrete Actions.
learner = ObliqueDTPolicy(clf, env)  #
# You can replace by DTPolicy(clf, env) for interpretable axis-parallel DTs.

# Start the imitation learning
interpret = Propel(learner, env)
interpret.fit_()

# Eval and save the best tree
final_tree_reward, _ = evaluate_policy(interpret._policy, env=env, n_eval_episodes=10)
print(final_tree_reward)
# Here you can replace pickle with joblib or cloudpickle
with open("tree_pendulum.pkl", "wb") as f:
    dump(interpret._policy.clf, f)

with open("tree_pendulum.pkl", "rb") as f:
    clf = load(f)

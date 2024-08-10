from interpreter import Interpreter
from interpreter import ObliqueDTPolicy, SB3Policy
from interpreter import parse_to_python

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym
from sklearn.tree import DecisionTreeClassifier
from huggingface_sb3 import load_from_hub

# Load the oracle policy
env = gym.make("Acrobot-v1")
model = PPO.load(
    load_from_hub(repo_id="sb3/ppo-Acrobot-v1", filename="ppo-Acrobot-v1.zip")
)
oracle = SB3Policy(model.policy)

learner = ObliqueDTPolicy(DecisionTreeClassifier(max_leaf_nodes=8), env)
interpret = Interpreter(oracle, learner, env)
interpret.fit(1e5)
print(evaluate_policy(model, env, n_eval_episodes=50))
print(evaluate_policy(interpret._policy, env, n_eval_episodes=50))
parse_to_python(
    interpret._policy.clf,
    env,
    feature_names=["cos_th1", "sin_th1", "cos_th2", "sin_th2", "vel_th1", "vel_th2"],
)

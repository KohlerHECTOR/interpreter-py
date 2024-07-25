from interpreter import SB3Policy, DTPolicy, ObliqueDTPolicy, Interpreter
import gymnasium as gym
from stable_baselines3 import PPO
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from rlberry.manager import (
    ExperimentManager,
    evaluate_agents,
)
from rlberry.envs import gym_make


def test_sb3_policy():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env)
    policy = SB3Policy(model.policy)
    s, _ = env.reset()
    policy.predict(s)


def test_sb3_policy_ctnuous_actions():
    env = gym.make("Pendulum-v1")
    model = PPO("MlpPolicy", env)
    policy = SB3Policy(model.policy)
    s, _ = env.reset()
    policy.predict(s)


def test_dt_policy():
    env = gym.make("CartPole-v1")
    clf = DecisionTreeClassifier(max_leaf_nodes=8)
    policy = DTPolicy(clf, env)
    s, _ = env.reset()
    policy.predict(s)


def test_dt_policy_ctnuous_actions():
    env = gym.make("Pendulum-v1")
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    policy = DTPolicy(clf, env)
    s, _ = env.reset()
    policy.predict(s)

def test_dt_policy_wrong_clf():
    env = gym.make("Acrobot-v1")
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    try:
        DTPolicy(clf, env)
    except AssertionError:
        pass

def test_dt_policy_ctnuous_actions_wrong_clf():
    env = gym.make("Pendulum-v1")
    clf = DecisionTreeClassifier(max_leaf_nodes=8)
    try:
        DTPolicy(clf, env)
    except AssertionError:
        pass


def test_oblique_dt_policy():
    env = gym.make("CartPole-v1")
    clf = DecisionTreeClassifier(max_leaf_nodes=8)
    policy = ObliqueDTPolicy(clf, env)
    s, _ = env.reset()
    policy.predict(s)


def test_oblique_dt_policy_ctnuous_actions():
    env = gym.make("Pendulum-v1")
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    policy = ObliqueDTPolicy(clf, env)
    s, _ = env.reset()
    policy.predict(s)


def test_interpreter():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeClassifier(max_leaf_nodes=8)
    learner = DTPolicy(clf, env)
    interpret = Interpreter(oracle, learner, env)
    interpret.fit(5)


def test_interpreter_oblique():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeClassifier(max_leaf_nodes=8)
    learner = ObliqueDTPolicy(clf, env)
    interpret = Interpreter(oracle, learner, env)
    interpret.fit(5)


def test_interpreter_ctnuous_actions():
    env = gym.make("Pendulum-v1")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    learner = DTPolicy(clf, env)
    interpret = Interpreter(oracle, learner, env)
    interpret.fit(3)


def test_interpreter_oblique_ctnuous_actions():
    env = gym.make("Pendulum-v1")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    learner = ObliqueDTPolicy(clf, env)
    interpret = Interpreter(oracle, learner, env)
    interpret.fit(3)
    interpret.policy(env.reset()[0])


def test_interpreter_oblique_ctnuous_actions_high_dim():
    env = gym.make("Ant-v4")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    learner = ObliqueDTPolicy(clf, env)
    interpret = Interpreter(oracle, learner, env)
    interpret.fit(3)
    interpret.policy(env.reset()[0])




def test_interpreter_ctnuous_actions_high_dim():
    env = gym.make("Ant-v4")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    learner = DTPolicy(clf, env)
    interpret = Interpreter(oracle, learner, env)
    interpret.fit(3)
    interpret.policy(env.reset()[0])


def test_interpreter_rlberry():
    env = gym.make("Ant-v4")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    learner = DTPolicy(clf, env)

    exp = ExperimentManager(
        agent_class=Interpreter,
        train_env=(gym_make, {"id": "Ant-v4"}),
        fit_budget=1e4,
        init_kwargs=dict(oracle=oracle, learner=learner),
        n_fit=2,
        seed=42,
    )
    exp.fit()

    _ = evaluate_agents(
        [exp], n_simulations=50, show=False
    )  # Evaluate the trained agent on

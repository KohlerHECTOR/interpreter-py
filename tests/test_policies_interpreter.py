from interpreter import SB3Policy, DTPolicy, ObliqueDTPolicy, Interpreter
import gymnasium as gym
from stable_baselines3 import PPO
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def test_sb3_policy():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env)
    policy = SB3Policy(model.policy)
    policy.generate_data(env, nb_data=100)
    s, _ = env.reset()
    policy.predict(s)


def test_sb3_policy_ctnuous_actions():
    env = gym.make("Pendulum-v1")
    model = PPO("MlpPolicy", env)
    policy = SB3Policy(model.policy)
    policy.generate_data(env, nb_data=100)
    s, _ = env.reset()
    policy.predict(s)


def test_dt_policy():
    env = gym.make("CartPole-v1")
    clf = DecisionTreeClassifier(max_leaf_nodes=8)
    policy = DTPolicy(clf, env)
    policy.generate_data(env, nb_data=100)
    s, _ = env.reset()
    policy.predict(s)


def test_dt_policy_ctnuous_actions():
    env = gym.make("Pendulum-v1")
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    policy = DTPolicy(clf, env)
    policy.generate_data(env, nb_data=100)
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
    policy.generate_data(env, nb_data=100)
    s, _ = env.reset()
    policy.predict(s)


def test_oblique_dt_policy_ctnuous_actions():
    env = gym.make("Pendulum-v1")
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    policy = ObliqueDTPolicy(clf, env)
    policy.generate_data(env, nb_data=100)
    s, _ = env.reset()
    policy.predict(s)


def test_interpreter():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeClassifier(max_leaf_nodes=8)
    tree_policy = DTPolicy(clf, env)
    interpret = Interpreter(oracle, tree_policy, env)
    interpret.train(5)


def test_interpreter_oblique():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeClassifier(max_leaf_nodes=8)
    tree_policy = ObliqueDTPolicy(clf, env)
    interpret = Interpreter(oracle, tree_policy, env)
    interpret.train(5)


def test_interpreter_ctnuous_actions():
    env = gym.make("Pendulum-v1")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    tree_policy = DTPolicy(clf, env)
    interpret = Interpreter(oracle, tree_policy, env)
    interpret.train(3)


def test_interpreter_oblique_ctnuous_actions():
    env = gym.make("Pendulum-v1")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    tree_policy = ObliqueDTPolicy(clf, env)
    interpret = Interpreter(oracle, tree_policy, env)
    interpret.train(3)


def test_interpreter_oblique_ctnuous_actions_high_dim():
    env = gym.make("Ant-v4")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    tree_policy = ObliqueDTPolicy(clf, env)
    interpret = Interpreter(oracle, tree_policy, env)
    interpret.train(3)


def test_interpreter_ctnuous_actions_high_dim():
    env = gym.make("Ant-v4")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    clf = DecisionTreeRegressor(max_leaf_nodes=8)
    tree_policy = DTPolicy(clf, env)
    interpret = Interpreter(oracle, tree_policy, env)
    interpret.train(3)
    interpret.get_best_tree_policy()

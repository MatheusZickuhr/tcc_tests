import gym
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from ple import PLE
from ple.games import FlappyBird
from rl.agents.dqn import DQNAgent
from rl.callbacks import  FileLogger
from rl.model_adapters import GymEnvAdapter, PleEnvAdapter
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from log_performance import log_performance


@log_performance(folder_path='training_data')
def main():
    env = PLE(FlappyBird(), display_screen=False, force_fps=True)
    env.init()
    env_adapter = PleEnvAdapter(env=env)

    print('actions number', env_adapter.get_n_actions())

    model = Sequential()
    model.add(Flatten(input_shape=(1, 8)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(env_adapter.get_n_actions(), activation='linear'))

    memory = SequentialMemory(limit=100000, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1,
                                  value_test=.5, nb_steps=1_000_000)

    dqn = DQNAgent(
        model=model,
        nb_actions=env_adapter.get_n_actions(),
        memory=memory,
        nb_steps_warmup=10,
        target_model_update=1e-2,
        policy=policy
    )
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    dqn.fit(
        env_adapter,
        nb_steps=1_750_000,
        visualize=True,
        verbose=2,
        callbacks=[FileLogger(filepath='training_data/log.json')]
    )

    dqn.save_weights('trained_model/model.h5f', overwrite=True)

    dqn.test(env_adapter, nb_episodes=5, visualize=True)

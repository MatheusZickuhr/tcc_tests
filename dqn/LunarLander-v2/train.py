import gym
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.callbacks import WandbLogger, FileLogger
from rl.model_adapters import GymEnvAdapter
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from log_performance import log_performance


@log_performance(folder_path='training_data')
def main():
    env_adapter = GymEnvAdapter(env=gym.make('LunarLander-v2'))

    model = Sequential()
    model.add(Flatten(input_shape=(1, 8)))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(env_adapter.get_n_actions(), activation='linear'))

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=env_adapter.get_n_actions(),
        memory=memory,
        nb_steps_warmup=10,
        target_model_update=1e-2,
        policy=policy
    )
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(
        env_adapter,
        nb_steps=1_000_000,
        visualize=True,
        verbose=2,
        callbacks=[FileLogger(filepath='training_data/log.json')]
    )

    dqn.save_weights('trained_model/model.h5f', overwrite=True)

    dqn.test(env_adapter, nb_episodes=5, visualize=True)

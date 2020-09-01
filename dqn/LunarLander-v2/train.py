from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger
from rl.model_adapters import GymEnvAdapter
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

from log_performance import log_performance


@log_performance(folder_path='training_data')
def main():
    env_adapter = GymEnvAdapter(env_name='LunarLander-v2', render=False)

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env_adapter.get_input_shape()))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(env_adapter.get_n_actions(), activation='linear'))

    memory = SequentialMemory(limit=1_000_000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=.1,
        value_test=.05,
        nb_steps=3_500_000
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=env_adapter.get_n_actions(),
        memory=memory,
        nb_steps_warmup=1000,
        policy=policy,
    )

    dqn.compile(Adam(lr=.00015), metrics=['mae'])

    dqn.fit(
        env_adapter,
        nb_steps=4_000_000,
        visualize=False,
        verbose=2,
        callbacks=[FileLogger(filepath='training_data/log.json'), ]
    )

    dqn.save_weights('trained_model/model.h5f', overwrite=True)
    dqn.test(env_adapter, nb_episodes=5, )

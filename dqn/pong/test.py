from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.model_adapters import PleEnvAdapter
from rl.memory import SequentialMemory

env_adapter = PleEnvAdapter(env_name='pong', render=False)

model = Sequential()
model.add(Flatten(input_shape=(1,) + env_adapter.get_input_shape()))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(env_adapter.get_n_actions(), activation='linear'))

memory = SequentialMemory(limit=1_000_000, window_length=1)

dqn = DQNAgent(
    model=model,
    nb_actions=env_adapter.get_n_actions(),
    memory=memory,
    nb_steps_warmup=1000,
)

dqn.compile(Adam(lr=.00015), metrics=['mae'])

dqn.load_weights('trained_model/model.h5f')
result = dqn.test(env_adapter, nb_episodes=500, )
file = open('trained_model/result.json', 'w+')
file.write('{"rewards":' + str(result.history["episode_reward"]) + '}')
file.close()

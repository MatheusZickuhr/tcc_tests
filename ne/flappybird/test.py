from neuroevolution_sandbox.agents.ne_agent import NeAgent
from neuroevolution_sandbox.env_adapters.ple_env_adapter import PleEnvAdapter
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter

env_adapter = PleEnvAdapter(env_name='flappybird', render=False, continuous=False)

agent = NeAgent(
    env_adapter=env_adapter,
    model_adapter=DefaultModelAdapter,
)

nn_config = (
    (env_adapter.get_input_shape(), 8, 'tanh'),
    (8, 'tanh'),
    (env_adapter.get_n_actions(), 'tanh')
)

agent.load('trained_model/model.json')

rewards = []
for i in range(500):
    reward = agent.play()
    rewards.append(reward)
    print(reward)

file = open('trained_model/result.json', 'w+')
file.write('{"rewards":' + str(rewards) + '}')
file.close()

from neuroevolution_sandbox.agents.ne_agent import NeAgent
from neuroevolution_sandbox.env_adapters.gym_env_adapter import GymEnvAdapter
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter

env_adapter = GymEnvAdapter(env_name='LunarLander-v2', render=False, continuous=False)

agent = NeAgent(
    env_adapter=env_adapter,
    model_adapter=DefaultModelAdapter,
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

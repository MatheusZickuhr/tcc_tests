from neuroevolution_sandbox.agents.neat_agent import NeatAgent
from neuroevolution_sandbox.env_adapters.ple_env_adapter import PleEnvAdapter


env_adapter = PleEnvAdapter(env_name='flappybird', render=False, continuous=False)
agent = NeatAgent(env_adapter=env_adapter, config_file_path='config.txt')
agent.load(file_path='trained_model/model')

rewards = []
for i in range(500):
    reward = agent.play()
    rewards.append(reward)
    print(reward)

file = open('trained_model/result.json', 'w+')
file.write('{"rewards":' + str(rewards) + '}')
file.close()

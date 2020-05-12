from neuroevolution_sandbox.agents.neat_agent import NeatAgent
from neuroevolution_sandbox.env_adapters.ple_env_adapter import PleEnvAdapter

env_adapter = PleEnvAdapter(env_name='pong', render=False, continuous=False)
agent = NeatAgent(env_adapter=env_adapter, config_file_path='config.txt')

agent.load(file_path='trained_model/model')
while 1: print(agent.play())

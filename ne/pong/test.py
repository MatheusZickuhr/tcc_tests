from neuroevolution_sandbox.agents.ne_agent import NeAgent
from neuroevolution_sandbox.env_adapters.ple_env_adapter import PleEnvAdapter
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter

env_adapter = PleEnvAdapter(env_name='pong', render=False, continuous=False)

agent = NeAgent(
    env_adapter=env_adapter,
    model_adapter=DefaultModelAdapter,
)

agent.load('trained_model/model.json')

while 1:
    print(agent.play())
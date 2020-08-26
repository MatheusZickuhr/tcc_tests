import json

file_data = open('results/dql/8x8/trained_model/result.json').read()

rewards = json.loads(file_data)['rewards']

print(sum(rewards))

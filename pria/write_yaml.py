import yaml

data = {
    'name': ['John Doe'],
    'age': 350,
    'address': {
        'street': '123 ain St',
        'city': 'Anytown',
        'state': 'CA',
        'zip': '12345'
    },
    'phone_numbers': ['555-1234', '555-5678']
}

# yaml_output = yaml.safe_load(data)

with open('./test.yaml', 'w') as file:
    yaml.dump(data, file,  default_flow_style=False)

# print(data['initial'])
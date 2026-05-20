from datasets import load_dataset

data_root = '~'

ob = load_dataset('math-ai/olympiadbench')['test']

def map_ob(example):
    item = {
        'data_source': 'math-ai/olympiadbench',
        'prompt': [{"role": "user", "content": example['question'].strip() + "\n\nLet's think step by step and output the final answer within \\boxed{}."}],
        'ability': 'math',
        'reward_model': {
            'style': 'rule',
            'ground_truth': example['final_answer'][0]
        },
        'extra_info': {
            'index': example['id'],
            'split': 'test',
            'question': example['question'],
        }
    }
    return item

ob = ob.map(map_ob, remove_columns=ob.column_names)
ob.to_parquet(f'{data_root}/rl-kl/verl/data/olympiadbench/test.parquet')

amc23 = load_dataset('math-ai/amc23')['test']

def map_amc23(example):
    item = {
        'data_source': 'math-ai/amc23',
        'prompt': [{"role": "user", "content": example['question'].strip() + "\n\nLet's think step by step and output the final answer within \\boxed{}."}],
        'ability': 'math',
        'reward_model': {
            'style': 'rule',
            'ground_truth': example['answer']
        },
        'extra_info': {
            'index': example['id'],
            'split': 'test',
            'question': example['question'],
        }
    }
    return item

amc23 = amc23.map(map_amc23, remove_columns=amc23.column_names)
amc23.to_parquet(f'{data_root}/rl-kl/verl/data/amc23/test.parquet')

minerva_math = load_dataset('math-ai/minervamath')['test']

def map_minerva_math(example, idx):
    item = {
        'data_source': 'math-ai/minervamath',
        'prompt': [{"role": "user", "content": example['question'].strip() + "\n\nLet's think step by step and output the final answer within \\boxed{}."}],
        'ability': 'math',
        'reward_model': {
            'style': 'rule',
            'ground_truth': example['answer']
        },
        'extra_info': {
            'index': idx,
            'split': 'test',
            'question': example['question'],
        }
    }
    return item

minerva_math = minerva_math.map(map_minerva_math, with_indices=True, remove_columns=minerva_math.column_names)
minerva_math.to_parquet(f'{data_root}/rl-kl/verl/data/minerva_math/test.parquet')

print(ob[0])
print('\n\n')
print(amc23[0])
print('\n\n')
print(minerva_math[0])
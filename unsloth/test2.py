from datasets import load_dataset
ds1 = load_dataset('CrowdMind/sft-tool-use-collection', split='train')
print('Dataset 1 loaded:', ds1.column_names)
print('First row keys:', ds1[0].keys())
print('First row data:', ds1[0])
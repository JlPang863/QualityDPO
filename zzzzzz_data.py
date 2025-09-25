# from datasets import load_dataset, Dataset, DatasetDict


# raw_dataset = load_dataset("jlpang888/ultrafeedback_with_learning_order")['train']



# # raw_dataset = load_dataset("json", data_files="ultrafeedback_with_learning_order.json")['train']
# test_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")['test_prefs']

# embedding_distance_sorted_dataset = raw_dataset.sort(f"embedding_distance", reverse=True)

# embeding_value = embedding_distance_sorted_dataset[int(len(embedding_distance_sorted_dataset) * 0.9)]['embedding_distance']

# # 先删掉原有的 score_chosen
# embedding_distance_sorted_dataset = embedding_distance_sorted_dataset.remove_columns(["score_chosen"])



# print(f"embeeding value: {embeding_value}")
# # 用 embedding_distance 替换成新的 score_chosen，并新增 score_rejected=0
# def transform(example):
#     example["score_chosen"] = example["embedding_distance"]
#     example["score_rejected"] = 0
#     return example

# embedding_distance_sorted_dataset = embedding_distance_sorted_dataset.map(transform)




# embedding_distance_sorted_dataset = embedding_distance_sorted_dataset.remove_columns([column_name for column_name in embedding_distance_sorted_dataset.column_names if column_name not in test_dataset.column_names ])

# import pdb;pdb.set_trace()


# new_dataset = DatasetDict({
#     'train': embedding_distance_sorted_dataset,
#     'test': test_dataset
# })

# new_dataset.push_to_hub("jlpang888/ultrafeedback_sorted_embedding_distance_new")



################################################################################################################
################################################################################################################
# from datasets import load_dataset, Dataset, DatasetDict


# raw_dataset = load_dataset("jlpang888/ultrafeedback_sorted_external_reward_new")

# raw_train_dataset, raw_test_dataset = raw_dataset['train'], raw_dataset['test']

# raw_train_dataset = raw_train_dataset.remove_columns(["score_chosen", "score_rejected"])
# raw_test_dataset = raw_test_dataset.remove_columns(["score_chosen", "score_rejected"])



# raw_train_dataset = raw_train_dataset.rename_column("reward_score_chosen", "score_chosen")
# raw_train_dataset = raw_train_dataset.rename_column("reward_score_rejected", "score_rejected")


# raw_test_dataset = raw_test_dataset.rename_column("reward_score_chosen", "score_chosen")
# raw_test_dataset = raw_test_dataset.rename_column("reward_score_rejected", "score_rejected")

# reward_score_threshold = raw_train_dataset[int(len(raw_train_dataset) * 0.9)]['score_chosen']
# print(f"reward_score_threshold value: {reward_score_threshold}")

# new_dataset = DatasetDict({
#     'train': raw_train_dataset,
#     'test': raw_test_dataset
# })


# import pdb;pdb.set_trace()
# new_dataset.push_to_hub("jlpang888/ultrafeedback_sorted_external_reward_new_0923")

################################################################################################################
################################################################################################################

# from datasets import load_dataset, Dataset, DatasetDict


# raw_dataset = load_dataset("jlpang888/ultrafeedback_with_learning_order")['train']




# # raw_dataset = load_dataset("json", data_files="ultrafeedback_with_learning_order.json")['train']
# test_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")['test_prefs']

# sorted_dataset = raw_dataset.sort(f"llama_learning_order", reverse=False)

# threshold = sorted_dataset[int(len(sorted_dataset) * 0.9)]['llama_learning_order']

# # 先删掉原有的 score_chosen
# sorted_dataset = sorted_dataset.remove_columns(["score_chosen"])



# print(f"threshold value: {threshold}")
# # 用 embedding_distance 替换成新的 score_chosen，并新增 score_rejected=0
# def transform(example):
#     example["score_chosen"] = example["llama_learning_order"]
#     example["score_rejected"] = 0
#     return example

# sorted_dataset = sorted_dataset.map(transform)
# sorted_dataset = sorted_dataset.remove_columns([column_name for column_name in sorted_dataset.column_names if column_name not in test_dataset.column_names ])


# new_dataset = DatasetDict({
#     'train': sorted_dataset,
#     'test': test_dataset
# })

# new_dataset.push_to_hub("jlpang888/ultrafeedback_sorted_llama_learning_order_new_0923")


################################################################################################################
################################################################################################################
from datasets import load_dataset, Dataset, DatasetDict

org_dataset= load_dataset('jlpang888/ultrafeedback_sorted_llama_learning_order')['train']

nwe_dataset = load_dataset("jlpang888/ultrafeedback_sorted_llama_learning_order_new_0923")['train']


import pdb;pdb.set_trace()
# from datasets import load_dataset, Dataset, DatasetDict


# raw_dataset = load_dataset("jlpang888/ultrafeedback_with_learning_order")['train']

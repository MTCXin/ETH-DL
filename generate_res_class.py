import json

# Step 1: Read the res.json file
with open('./result_black_simbav2.json', 'r') as file:
    items = json.load(file)

classified_items = {}


# Step 2: Process the values (treat 0 as maximum value)
max_value_l2 = 15
max_value_q = 24000
ratio = [0,0,0]
for item_id, value in items.items():
    if value['l2_norm'] == 0:
        items[item_id]["l2_norm"] = max_value_l2
        items[item_id]["queries:"] = max_value_q
    if 0 < value['l2_norm'] <= 5:
        classified_items[item_id] = {"l2_norm_class":0}
        ratio[0] += 1
    elif 5 < value['l2_norm'] <= 10:
        classified_items[item_id] = {"l2_norm_class":1}
        ratio[1] += 1
    elif 10 < value['l2_norm'] <= 15:
        classified_items[item_id] = {"l2_norm_class":2}
        ratio[2] += 1
    # else:
    #     classified_items[item_id] = {"l2_norm_class":3}
    #     ratio[3] += 1
    item_id = item_id.replace('../','')
ratio = [r/sum(ratio) for r in ratio]
print(ratio)




# # Step 3: Sort the items based on their values
# sorted_items = sorted(items.items(), key=lambda x: x[1]["l2_norm"])

# # Step 4: Divide the sorted items into 4 classes
# class_thresholds = len(sorted_items) // 4
# classified_items = {}
# for index, (item_id, _) in enumerate(sorted_items):
#     classified_items[item_id] = {"l2_norm_class":index // class_thresholds}
#     if classified_items[item_id]["l2_norm_class"] > 3:  # Ensure no class index goes above 3
#         classified_items[item_id]["l2_norm_class"] = 3

# # Step 3: Sort the items based on their values
# sorted_items = sorted(items.items(), key=lambda x: x[1]["queries:"])

# # Step 4: Divide the sorted items into 4 classes
# class_thresholds = len(sorted_items) // 4
# for index, (item_id, _) in enumerate(sorted_items):
#     classified_items[item_id]["query_class"] = index // class_thresholds
#     if classified_items[item_id]["query_class"] > 3:  # Ensure no class index goes above 3
#         classified_items[item_id]["query_class"] = 3

# Step 5: Write the classified items to class.json
with open('result_black_simba_class.json', 'w') as file:
    json.dump(classified_items, file)

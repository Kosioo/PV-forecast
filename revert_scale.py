import os
import re

def revert_scaling(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Revert y_train
    content = re.sub(r"y_train = dataset_15min\[train_mask\]\['actual_power_kw'\]", 
                     r"y_train = dataset_15min[train_mask]['actual_power_kw'] / 1000.0", content)
    content = re.sub(r"y_train_1h = train_1h_df\['actual_power_kw'\]", 
                     r"y_train_1h = train_1h_df['actual_power_kw'] / 1000.0", content)
    content = re.sub(r"y_train = dataset_hist\['actual_power_kw'\]", 
                     r"y_train = dataset_hist['actual_power_kw'] / 1000.0", content)
                     
    # Revert base_preds_train
    content = re.sub(r"base_preds_train = base_model\.predict\(X_train_base\) \* CAPACITY_KWP", 
                     r"base_preds_train = base_model.predict(X_train_base)", content)
    content = re.sub(r"base_preds_train_1h = base_model\.predict\(X_train_base_1h\) \* CAPACITY_KWP", 
                     r"base_preds_train_1h = base_model.predict(X_train_base_1h)", content)
                     
    # Revert test predictions
    # evaluate_topola.py
    content = re.sub(r"pred_base_only_15 = preds_15_base \* CAPACITY_KWP", 
                     r"pred_base_only_15 = preds_15_base * 1000.0", content)
    content = re.sub(r"pred_native_15 = \(preds_15_base \* CAPACITY_KWP\) \+ preds_15_res", 
                     r"pred_native_15 = (preds_15_base + preds_15_res) * 1000.0", content)
                     
    content = re.sub(r"pred_base_only_1h = preds_1h_base \* CAPACITY_KWP", 
                     r"pred_base_only_1h = preds_1h_base * 1000.0", content)
    content = re.sub(r"pred_native_1h = \(preds_1h_base \* CAPACITY_KWP\) \+ preds_1h_res", 
                     r"pred_native_1h = (preds_1h_base + preds_1h_res) * 1000.0", content)
                     
    # generate_topola_schedule.py
    content = re.sub(r"pred_base_only = preds_base \* CAPACITY_KWP", 
                     r"pred_base_only = preds_base * 1000.0", content)
    content = re.sub(r"pred_native = \(preds_base \* CAPACITY_KWP\) \+ preds_res", 
                     r"pred_native = (preds_base + preds_res) * 1000.0", content)

    with open(filepath, 'w') as f:
        f.write(content)

revert_scaling("evaluate_topola.py")
revert_scaling("generate_topola_schedule.py")
revert_scaling("evaluate_topola_base.py")
revert_scaling("generate_topola_schedule_base.py")

print("Reverted to original scaling.")

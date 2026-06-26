import os
import re

def fix_scaling(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix y_train scaling: actual_power_kw / 1000.0 -> actual_power_kw
    content = re.sub(r"y_train = dataset_15min\[train_mask\]\['actual_power_kw'\] / 1000\.0", 
                     r"y_train = dataset_15min[train_mask]['actual_power_kw']", content)
    content = re.sub(r"y_train_1h = train_1h_df\['actual_power_kw'\] / 1000\.0", 
                     r"y_train_1h = train_1h_df['actual_power_kw']", content)
    content = re.sub(r"y_train = dataset_hist\['actual_power_kw'\] / 1000\.0", 
                     r"y_train = dataset_hist['actual_power_kw']", content)
                     
    # Fix base_preds_train scaling: base_preds_train = base_model.predict(...) * CAPACITY_KWP
    content = re.sub(r"base_preds_train = base_model\.predict\(X_train_base\)", 
                     r"base_preds_train = base_model.predict(X_train_base) * CAPACITY_KWP", content)
    content = re.sub(r"base_preds_train_1h = base_model\.predict\(X_train_base_1h\)", 
                     r"base_preds_train_1h = base_model.predict(X_train_base_1h) * CAPACITY_KWP", content)
                     
    # Fix test predictions scaling: preds_15_base * 1000.0 -> preds_15_base * CAPACITY_KWP
    # evaluate_topola.py
    content = re.sub(r"pred_base_only_15 = preds_15_base \* 1000\.0", 
                     r"pred_base_only_15 = preds_15_base * CAPACITY_KWP", content)
    content = re.sub(r"pred_native_15 = \(preds_15_base \+ preds_15_res\) \* 1000\.0", 
                     r"pred_native_15 = (preds_15_base * CAPACITY_KWP) + preds_15_res", content)
                     
    content = re.sub(r"pred_base_only_1h = preds_1h_base \* 1000\.0", 
                     r"pred_base_only_1h = preds_1h_base * CAPACITY_KWP", content)
    content = re.sub(r"pred_native_1h = \(preds_1h_base \+ preds_1h_res\) \* 1000\.0", 
                     r"pred_native_1h = (preds_1h_base * CAPACITY_KWP) + preds_1h_res", content)
                     
    # generate_topola_schedule.py
    content = re.sub(r"pred_base_only = preds_base \* 1000\.0", 
                     r"pred_base_only = preds_base * CAPACITY_KWP", content)
    content = re.sub(r"pred_native = \(preds_base \+ preds_res\) \* 1000\.0", 
                     r"pred_native = (preds_base * CAPACITY_KWP) + preds_res", content)

    with open(filepath, 'w') as f:
        f.write(content)

fix_scaling("evaluate_topola.py")
fix_scaling("generate_topola_schedule.py")
fix_scaling("evaluate_topola_base.py")
fix_scaling("generate_topola_schedule_base.py")

print("Scaling fixed in all 4 files.")

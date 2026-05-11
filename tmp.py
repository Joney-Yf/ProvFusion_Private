from math import sqrt

total_final_TP, total_final_FP, total_final_TN, total_final_FN=2,4,1470506,112

precision = total_final_TP / (total_final_TP + total_final_FP + 1e-12)
recall = total_final_TP / (total_final_TP + total_final_FN + 1e-12)
accuracy = (total_final_TP + total_final_TN) / (total_final_TP + total_final_FP + total_final_FN + total_final_TN + 1e-12)
f1 = 2 * precision * recall / (precision + recall + 1e-12)
mcc = (total_final_TP * total_final_TN - total_final_FP * total_final_FN) / sqrt( 
    (total_final_TP + total_final_FP + 1e-12) *
    (total_final_TP + total_final_FN + 1e-12) *
    (total_final_TN + total_final_FP + 1e-12) *
    (total_final_TN + total_final_FN + 1e-12)
)


print(f"[Final Results]")
print(f"TP={total_final_TP}, FP={total_final_FP}, FN={total_final_FN}, TN={total_final_TN}")
print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}, MCC={mcc:.4f}")

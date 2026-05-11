normalizations=("min_max" "z_score" "robust" "quantile")
# fusion threshold method
# methods=("max" "l2_norm" "max_two" "dynamic_max_two" "dynamic_three")
## Probability model
# methods=('mvg' 'gmm')
# n_components=(1 3 5 7 10 12 15)
## Density-based model
# methods=('knn')
## Boundary-based model
methods=('ocsvm')
## Reconstruction-based model
# methods=("autoencoder")
n_components=(100)
lrs=(1e-3  1e-4 1e-5 1e-6 1e-7)

# 遍历所有 normalization 和 method 组合
for norm in "${normalizations[@]}"; do
    for meth in "${methods[@]}"; do
        for n in "${n_components[@]}"; do
            for lr in "${lrs[@]}"; do
                echo "Running python systematic_threshold_study.py with normalization=$norm, method=$meth, n_components=$n"
                CUDA_VISIBLE_DEVICES=1 python systematic_threshold_study.py --normalization "$norm" --method "$meth" --n_components "$n" --lr "$lr" > my_ablation_analysis/$norm\_$meth\_$n\_$lr\.log
                if [ $? -ne 0 ]; then
                    echo "Error running with normalization=$norm, method=$meth, n_components=$n, lr=$lr"
                    exit 1
                fi
            done
        done
    done
done

echo "All combinations completed."
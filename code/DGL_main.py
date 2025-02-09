import os

def run_dglke_training(model_name, dataset, data_path, output_dir, gpu_ids, batch_size=2000, neg_sample_size=200, hidden_dim=400, gamma=19.9, lr=0.2, max_step=24000, log_interval=100, batch_size_eval=16, regularization_coef=1e-9, num_thread=1, num_proc=6):
    # Construct the command
    command = f"""
    DGLBACKEND=pytorch dglke_train \\
    --model_name {model_name} \\
    --dataset {dataset} \\
    --data_path {data_path} \\
    --format raw_udd_hrt \\
    --data_files train.txt valid.txt test.txt \\
    --batch_size {batch_size} \\
    --neg_sample_size {neg_sample_size} \\
    --hidden_dim {hidden_dim} \\
    --gamma {gamma} \\
    --lr {lr} \\
    --max_step {max_step} \\
    --log_interval {log_interval} \\
    --batch_size_eval {batch_size_eval} \\
    -adv -a 1.0 \\
    --regularization_coef={regularization_coef} \\
    --test --num_thread {num_thread} \\
    --num_proc {num_proc} \\
    --gpu {' '.join(map(str, gpu_ids))}
    """
    os.system(command)
    print(f"Training for model {model_name} completed and saved in {output_dir}")

def run_dglke_evaluation(model_name, dataset, data_path, output_dir, gpu_ids, hidden_dim=400, gamma=19.9, batch_size_eval=16):
    command = f"""
    DGLBACKEND=pytorch dglke_eval \\
    --model_name {model_name} \\
    --dataset {dataset} \\
    --data_path {data_path} \\
    --format raw_udd_hrt \\
    --data_files train.txt valid.txt test.txt \\
    --hidden_dim {hidden_dim} \\
    --gamma {gamma} \\
    --batch_size_eval {batch_size_eval} \\
    --gpu {' '.join(map(str, gpu_ids))} \\
    --model_path {output_dir}/{model_name}_best_model.pt
    """
    os.system(command)
    print(f"Evaluation for model {model_name} completed and results saved in {output_dir}")


if __name__ == "__main__":
    model_name = "TransE"
    dataset = "WJoconde"
    data_path = "./dgl-ke_data/"
    output_dir = "./ckpts"
    gpu_ids = [0, 1, 2]
    
    # Run training
    run_dglke_training(model_name, dataset, data_path, output_dir, gpu_ids)
    
    # Run evaluation
    run_dglke_evaluation(model_name, dataset, data_path, output_dir, gpu_ids)












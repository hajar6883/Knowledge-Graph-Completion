import argparse
import os
import torch
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.models import TransE, DistMult, ComplEx, RESCAL, RotatE, TuckER, TransH, TransD, TransR
from pykeen.evaluation import RankBasedEvaluator
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
    
def save_metrics(metrics_dict, save_path, existing_df=None):
    metrics_df = pd.DataFrame([metrics_dict])
    
    if existing_df is not None:
        combined_df = pd.concat([existing_df, metrics_df], ignore_index=True, sort=False)
        combined_df = combined_df.reindex(columns=existing_df.columns.union(metrics_df.columns, sort=False))
    else:
        combined_df = metrics_df

    combined_df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}")
    return combined_df


def run_pipeline(model, training, testing, validation, device, nb_epochs):
    # Run the pipeline to train the model
    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model=model,
        stopper='early',
        evaluator=RankBasedEvaluator,
        training_loop='sLCWA',
        negative_sampler='basic',
        epochs=nb_epochs,
        model_kwargs={'embedding_dim': 500},
        loss='MarginRankingLoss',
        random_seed=420,
        device=device
    )

    # Get the best model from the training result
    best_model = result.model
    
    # Evaluate the best model on the testing set
    evaluator = RankBasedEvaluator()
    evaluation_results = evaluator.evaluate(
        model=best_model,
        mapped_triples=testing.mapped_triples,
        additional_filter_triples=[training.mapped_triples, validation.mapped_triples],
        device=device
    )

    # Extract metrics
    new_best_mrr = evaluation_results.get_metric("mean_reciprocal_rank")
    hits_at_1 = evaluation_results.get_metric("hits_at_1")
    hits_at_3 = evaluation_results.get_metric("hits_at_3")
    hits_at_10 = evaluation_results.get_metric("hits_at_10")

    # Determine model name and save path
    model_name = model.__name__ if isinstance(model, type) else model.__class__.__name__
    model_save_path = f"./pykeen_models/{model_name}_best_model.pt"
    print("Model save path:", model_save_path)

    save_new_model = True
    if os.path.exists(model_save_path):
        print('Previous model found, evaluating...')
        prev_model = model(triples_factory=training, embedding_dim=500) 
        prev_model.load_state_dict(torch.load(model_save_path))
        prev_model.to(device)

        prev_evaluation_results = evaluator.evaluate(
            model=prev_model,
            mapped_triples=testing.mapped_triples,
            additional_filter_triples=[training.mapped_triples, validation.mapped_triples],
            device=device
        )
        prev_best_mrr = prev_evaluation_results.get_metric("mean_reciprocal_rank")
        
        if prev_best_mrr >= new_best_mrr:
            print(f"Existing model has better or equal performance (MRR: {prev_best_mrr}), not saving new model.")
            save_new_model = False
    else:
        print("No previous model found. Saving the current model.")
   
    if save_new_model:
        torch.save(best_model.state_dict(), model_save_path)
        print(f"Model {model_name} saved to {model_save_path} with MRR: {new_best_mrr}")

    # Collect metrics
    metrics_dict = {
        'Model': model_name,
        'Best_MRR': new_best_mrr,
        'embedding_dim': 500,  
        'nb_epochs': nb_epochs,
        'Hits@1': hits_at_1,
        'Hits@3': hits_at_3,
        'Hits@10': hits_at_10
    }

    return metrics_dict

def run_study(model, training, testing, validation, device, checkpoint_dir, nb_epochs=100, n_trials=10):
    print(f'Starting a new study on model: {model}...')
    hpo_result = hpo_pipeline(
        model=model,
        training=training,
        validation=validation,
        testing=testing,
        stopper='early',
        epochs=nb_epochs,
        model_kwargs={
            'embedding_dim': 512},
        evaluator_kwargs={
            'filtered': True
        },
        n_trials=n_trials,
        device=device
    )

    best_trial = hpo_result.study.best_trial
    best_model_kwargs = best_trial.params
    model_hyperparameters = {
        key.split('.')[-1]: value for key, value in best_model_kwargs.items()
        if key.startswith('model.')
    }

    best_model = model(
        triples_factory=training,  
        **model_hyperparameters  
    )

    model_name = model.__class__.__name__
    model_save_path = f"{checkpoint_dir}/{model_name}_best_model.pt"
    torch.save(best_model.state_dict(), model_save_path)
    print(f"Best {model_name} model saved at {model_save_path}")

    evaluator = RankBasedEvaluator(filtered=True)
    metric_results = evaluator.evaluate(
        model=best_model,
        mapped_triples=testing.mapped_triples,
        additional_filter_triples=validation.mapped_triples,
        device=device
    )
    
    best_mrr = metric_results.get_metric('mean_reciprocal_rank')
    best_hits_at_1 = metric_results.get_metric('hits@1')
    best_hits_at_3 = metric_results.get_metric('hits@3')
    best_hits_at_10 = metric_results.get_metric('hits@10')
    
    best_epoch = best_trial.number
    best_loss = hpo_result.study.best_value

    model_scoring_fct_norm = best_model_kwargs.get('model.scoring_fct_norm')
    loss_margin = best_model_kwargs.get('loss.margin')
    optimizer_lr = best_model_kwargs.get('optimizer.lr')
    negative_sampler_num_negs_per_pos = best_model_kwargs.get('negative_sampler.num_negs_per_pos')
    training_batch_size = best_model_kwargs.get('training.batch_size')

    metrics_dict= {
        'Model': model_name,
        'Best_MRR': best_mrr,
        'Best_Hits@1': best_hits_at_1,
        'Best_Hits@3': best_hits_at_3,
        'Best_Hits@10': best_hits_at_10,
        'Best_Epoch': best_epoch,
        'Best_Loss': best_loss,
        'model_scoring_fct_norm': model_scoring_fct_norm,
        'loss_margin': loss_margin,
        'optimizer_lr': optimizer_lr,
        'negative_sampler_num_negs_per_pos': negative_sampler_num_negs_per_pos,
        'training_batch_size': training_batch_size,
        'nb_epochs': nb_epochs
    } 

    return metrics_dict

    
    
def main():
    
    parser = argparse.ArgumentParser(description="Run a knowledge graph embedding pipeline with PyKEEN.")
    parser.add_argument('--data_file_path', type=str, default="./PyG_data/cleaned_triples.tsv", help="Path to the data file.")
    parser.add_argument('--checkpoint_dir', type=str, default="./pykeen_models", help="Directory to save model checkpoints.")
    parser.add_argument('--metrics_save_path', type=str, default="./pykeen_models/benchmark_metrics.csv", help="Path to save the metrics.")
    parser.add_argument('--model', type=str, default="TuckER", help="Name of the model to run.")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--n_trials', type=int, default=10, help="Number of HPO trials.")
    parser.add_argument('--use_hpo', action='store_true', help="Flag to run HPO instead of a standard pipeline.")
    
    args = parser.parse_args()

    # Handle model selection
    model_class = {
        "TransE": TransE,
        "TransH": TransH,
        "TransD": TransD,
        "TransR": TransR,
        "DistMult": DistMult,
        "ComplEx": ComplEx,
        "RESCAL": RESCAL,
        "RotatE": RotatE,
        "TuckER": TuckER
        
    }.get(args.model, TuckER)  
    
 

    data = pd.read_csv(args.data_file_path, header=None, sep='\t' ,names=["source", "type", "target"])
    
    tf = TriplesFactory.from_labeled_triples(
        data[["source", "type", "target"]].values,
        create_inverse_triples=False,
        entity_to_id=None,
        relation_to_id=None,
        compact_id=False,
        filter_out_candidate_inverse_relations=True,
        metadata=None,
    )

    training, testing, validation = tf.split([.8, .1, .1], random_state=3031781928)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.use_hpo:
        metrics_dict = run_study(model_class, training, testing, validation, device, args.checkpoint_dir, nb_epochs=args.num_epochs, n_trials=args.n_trials)
    else:
        metrics_dict = run_pipeline(model_class, training, testing, validation, device,nb_epochs=args.num_epochs)


    existing_df = pd.read_csv(args.metrics_save_path) if os.path.exists(args.metrics_save_path) else None
    save_metrics(metrics_dict, args.metrics_save_path, existing_df)


if __name__=="__main__":
    main()

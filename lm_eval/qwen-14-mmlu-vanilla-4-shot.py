import evaluator
from loggers import EvaluationTracker


def main():
    eval_tracker = EvaluationTracker("results/")

    results = evaluator.simple_evaluate(
        model="vllm",
        model_args="pretrained=Qwen/Qwen2.5-14B-Instruct,max_length=128000,max_gen_toks=8192,tensor_parallel_size=2,data_parallel_size=1",
        tasks="mmlu_flan_n_shot_generative",
        batch_size="auto",
        num_fewshot=4,
        device="cuda",
        use_cache=None,
        limit=None,
        check_integrity=False,
        write_out=True,
        log_samples=True,
        evaluation_tracker=eval_tracker,
        system_instruction=None,
        apply_chat_template=True,
        fewshot_as_multiturn=False,
        task_manager=None,
        verbosity="INFO",
        predict_only=False,
        random_seed=1,
        numpy_random_seed=1,
        torch_random_seed=1,
        fewshot_random_seed=1,
    )
    eval_tracker.save_results_aggregated(results=results, samples=results["samples"])


if __name__ == "__main__":
    main()

import evaluator
from loggers import EvaluationTracker


def main():
    eval_tracker = EvaluationTracker("results/")

    results = evaluator.simple_evaluate(
        model="vllm-tot",
        model_args="pretrained=BEE-spoke-data/smol_llama-101M-GQA,max_length=1024",
        tasks="gsm8k",
        num_fewshot=0,
        device="cuda",
        use_cache=None,
        limit=1,
        check_integrity=False,
        write_out=True,
        log_samples=True,
        evaluation_tracker=eval_tracker,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        gen_kwargs="method_generate=propose,method_evaluate=value,method_select=greedy,steps=3,n_generate_sample=3,n_evaluate_sample=3,n_select_sample=5",
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

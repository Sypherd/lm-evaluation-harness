import evaluator
from loggers import EvaluationTracker

import lm_eval


def main():
    lm = lm_eval.api.registry.get_model("vllm").create_from_arg_string(
        "pretrained=Qwen/Qwen2.5-32B-Instruct,max_length=128000,max_gen_toks=8192,tensor_parallel_size=2,data_parallel_size=1",
        {
            "batch_size": "auto",
            "max_batch_size": None,
            "device": "cuda",
        },
    )

    for fewshot in range(0, 9):
        eval_tracker = EvaluationTracker("results/ablation/qwen-32-cot")
        results = evaluator.simple_evaluate(
            model=lm,
            model_args="",
            tasks="gsm8k_cot",
            num_fewshot=fewshot,
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
        eval_tracker.general_config_tracker.model_source = None
        eval_tracker.save_results_aggregated(
            results=results, samples=results["samples"]
        )


if __name__ == "__main__":
    main()

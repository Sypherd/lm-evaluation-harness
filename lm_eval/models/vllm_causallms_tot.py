import copy
import itertools
import re
from typing import List

import numpy as np
from vllm.outputs import RequestOutput

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.prompts import PROMPT_REGISTRY
from lm_eval.utils import eval_logger


@register_model("vllm-tot")
class VLLM_ToT(VLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_tokens_count = 0
        self.out_tokens_count = 0
        self.value_cache = {}
        self.task = "gsm8k"

    def get_samples(self, context, y, n_samples, max_gen_toks, until):
        prompt = (
            PROMPT_REGISTRY["tree-of-thought"][self.task]["cot-prompt"].format(
                input=context
            )
            + y
        )
        context_encoding = self.prepare_context(prompt, max_gen_toks)

        # TODO check when max_gen_toks is None
        samples = self._model_generate(
            requests=context_encoding,
            generate=True,
            max_tokens=max_gen_toks,
            stop=until,
            n=n_samples,
        )
        self.count_out_tokens(samples)

        return [y + s.text for s in samples[0].outputs]

    def get_proposals(self, context, y, max_gen_toks, until):
        propose_prompt = PROMPT_REGISTRY["tree-of-thought"][self.task][
            "propose-prompt"
        ].format(input=context)
        context_encoding = self.prepare_context(propose_prompt, max_gen_toks)

        proposals = self._model_generate(
            requests=context_encoding,
            generate=True,
            max_tokens=max_gen_toks,
            stop=until,
            # n=1,
        )
        self.count_out_tokens(proposals)
        proposals = proposals[0].outputs[0].text.split("\n")

        return [y + _ + "\n" for _ in proposals]

    @staticmethod
    def value_outputs_unwrap(y: str, value_outputs: list[RequestOutput]) -> float:
        if len(y.strip().split("\n")) == 4 and "answer" not in y.lower():
            return 0
        value_names = [vo.text.split("\n")[-1] for vo in value_outputs[0].outputs]
        value_map = {"impossible": 0.001, "likely": 1, "sure": 20}  # TODO: ad hoc
        value = sum(
            value * value_names.count(name) for name, value in value_map.items()
        )  # brittle check
        return value

    def get_value(self, context, y, n_samples, max_gen_toks, until, cache_value=True):
        value_prompt = PROMPT_REGISTRY["tree-of-thought"][self.task][
            "value-prompt"
        ].format(input=context)

        if cache_value and value_prompt in self.value_cache:
            return self.value_cache[value_prompt]

        context_encoding = self.prepare_context(value_prompt, max_gen_toks)

        value_outputs = self._model_generate(
            requests=context_encoding,
            generate=True,
            max_tokens=max_gen_toks,
            stop=until,
            n=n_samples,
        )
        self.count_out_tokens(value_outputs)
        value = self.value_outputs_unwrap(y, value_outputs)
        if cache_value:
            self.value_cache[value_prompt] = value

        return value

    def get_values(self, context, ys, n_samples, max_gen_toks, until, cache_value=True):
        values = []
        local_value_cache = {}
        for y in ys:  # each partial output
            if y in local_value_cache:  # avoid duplicate candidates
                value = 0
            else:
                value = self.get_value(
                    context, y, n_samples, max_gen_toks, until, cache_value=cache_value
                )
                local_value_cache[y] = value
            values.append(value)
        return values

    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"  # brittle
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f"vote no match: {[vote_output]}")
        return vote_results

    def get_votes(self, context, ys, n_samples, max_gen_toks, until):
        vote_prompt = PROMPT_REGISTRY["tree-of-thought"][self.task][
            "vote-prompt"
        ].format(input=context)
        context_encoding = self.prepare_context(vote_prompt, max_gen_toks)

        vote_outputs = self._model_generate(
            requests=context_encoding,
            generate=True,
            max_tokens=max_gen_toks,
            stop=until,
            n=n_samples,
        )
        self.count_out_tokens(vote_outputs)
        values = self.vote_outputs_unwrap(vote_outputs, len(ys))

        return values

    def prepare_context(self, context, max_gen_toks):
        context_encoding = self.tok_encode(
            context, add_special_tokens=self.add_bos_token
        )
        # set the max length in tokens of inputs ("context_enc")
        # max len for inputs = max length, minus room to generate the max new tokens
        max_ctx_len = self.max_length - max_gen_toks
        if isinstance(context_encoding[0], list):
            context_encoding = [x[:max_ctx_len] for x in context_encoding]
            self.in_tokens_count += sum(len(x) for x in context_encoding)
        else:
            context_encoding = context_encoding[:max_ctx_len]
            self.in_tokens_count += len(context_encoding)

        return context_encoding

    def count_out_tokens(self, outputs):
        for output in outputs[0].outputs:
            self.out_tokens_count += len(output.token_ids)

    def generate_until(self, requests: list[Instance], to_print=True) -> List[str]:
        self.task = requests[0].task_name.split("_")[0]
        yys = []
        for request in requests:
            context, gen_kwargs = request.args

            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}"
                )
            # add EOS token to stop sequences
            eos = self.tokenizer.decode(self.eot_token_id)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            ys = [""]  # current output candidates
            infos = []
            for step in range(gen_kwargs["depth"]):
                # generation
                if gen_kwargs["method_generate"] == "sample":
                    new_ys = [
                        self.get_samples(
                            context,
                            y,
                            gen_kwargs["n_generate_sample"],
                            max_gen_toks,
                            until,
                        )
                        for y in ys
                    ]
                elif gen_kwargs["method_generate"] == "propose":
                    new_ys = [
                        self.get_proposals(context, y, max_gen_toks, until) for y in ys
                    ]
                else:
                    raise ValueError(
                        f"method {gen_kwargs['method_generate']} is not supported"
                    )
                new_ys = list(itertools.chain(*new_ys))
                ids = list(range(len(new_ys)))
                # evaluation
                if gen_kwargs["method_evaluate"] == "vote":
                    values = self.get_votes(
                        context,
                        new_ys,
                        gen_kwargs["n_evaluate_sample"],
                        max_gen_toks,
                        until,
                    )
                elif gen_kwargs["method_evaluate"] == "value":
                    values = self.get_values(
                        context,
                        new_ys,
                        gen_kwargs["n_evaluate_sample"],
                        max_gen_toks,
                        until,
                    )
                else:
                    raise ValueError(
                        f"method {gen_kwargs['method_evaluate']} is not supported"
                    )

                # selection
                if gen_kwargs["method_select"] == "sample":
                    ps = np.array(values) / sum(values)
                    select_ids = np.random.choice(
                        ids, size=gen_kwargs["n_select_sample"], p=ps
                    ).tolist()
                elif gen_kwargs["method_select"] == "greedy":
                    select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[
                        : gen_kwargs["n_select_sample"]
                    ]
                else:
                    raise ValueError(
                        f"method {gen_kwargs['method_select']} is not supported"
                    )
                select_new_ys = [new_ys[select_id] for select_id in select_ids]

                # log
                sorted_new_ys, sorted_values = zip(
                    *sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True)
                )
                eval_logger.debug(
                    f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n"
                )

                infos.append(
                    {
                        "step": step,
                        "context": context,
                        "ys": ys,
                        "new_ys": new_ys,
                        "values": values,
                        "select_new_ys": select_new_ys,
                    }
                )
                ys = select_new_ys

            yys.append(ys[0])  # See note
            # In https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/prompts/game24.py,
            # there's a final value prompt. Interestingly, there is no such valuation
            # for the crosswords game. I think just taking the highest value here makes sense

        return yys

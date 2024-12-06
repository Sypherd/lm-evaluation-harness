import ast
import os
from typing import Dict

from lm_eval import utils
from lm_eval.utils import eval_logger

# Prompt library.
# Stores prompts in a dictionary indexed by 2 levels:
# prompt category name, and prompt name.
# This allows us to access prompts
PROMPT_REGISTRY: Dict[str, Dict[str, str]] = {
    "qa-basic": {
        "question-newline-answer": "Question: {{question}}\nAnswer:",
        "q-newline-a": "Q: {{question}}\nA:",
    },
    "tree-of-thought": {
        "mmlu": {
            # CoT 8-shot for evaluation
            "cot-prompt": """{input}""",
            "propose-prompt": """Question: "Which of the following scenarios best exemplifies utilitarian reasoning as applied during the Industrial Revolution?"
Possible answers:
A. Enclosure of common lands to increase agricultural productivity despite widespread rural displacement.
B. Maintaining guild systems to protect artisanal craftsmen at the expense of economic innovation.
C. Supporting the abolition of child labor regardless of its impact on family incomes.
D. Advocating for personal freedom over societal welfare in debates on factory safety regulations.
Proposals:
- Option A: Assess whether the increased agricultural productivity from enclosures had long-term societal benefits, such as supporting population growth and urbanization.
- Option B: Investigate whether maintaining guild systems directly conflicts with utilitarian principles by prioritizing the welfare of a small group over the potential economic innovation benefiting a larger population.
- Option C: Explore whether abolition of child labor had utilitarian support based on potential long-term societal benefits, such as healthier, more educated adults contributing to the economy.
- Option D: Determine whether prioritizing personal freedom over societal welfare aligns with the utilitarian emphasis on collective well-being.

Question: {question}
Possible answers: {choices}
Proposals:""",
            "value-prompt": """{input}""",
            "vote-prompt": """{input}""",
        },
        "gpqa": {
            # CoT 8-shot for evaluation
            "cot-prompt": """{input}""",
            "propose-prompt": """QUestion: In the context of the double-slit experiment, which phenomenon demonstrates the wave-particle duality of light?
Possible answers:
A) The diffraction pattern observed when light passes through a single slit. 
B) The interference pattern formed when light passes through two slits and is observed on a screen. 
C) The photoelectric effect, where light causes the emission of electrons from a metal surface. 
D) The gravitational lensing of light as it passes near a massive object.
Proposals:
- The question is about wave-particle duality. What is wave-particle duality? It refers to light behaving both as a wave and as a particle under different circumstances.
- What happens when light passes through two slits? It creates an interference pattern on a screen, suggesting wave behavior.
- Is the interference pattern seen in the double-slit experiment a result of light acting like a wave? Yes, it is, since interference is a property of waves.
- The photoelectric effect (Option C) demonstrates the particle nature of light, but does it relate to the double-slit experiment? No, the photoelectric effect isn’t part of that experiment.
- If the double-slit experiment shows light can create an interference pattern, does this alone demonstrate wave-particle duality? No, but it shows the wave behavior, and later, when observed as photons, the particle nature is revealed.

Question: {question}
Possible answers: {choices}
Proposals:""",
            "value-prompt": """{input}""",
            "vote-prompt": """{input}""",
        },
        "gsm8k": {
            # CoT 8-shot for evaluation
            "cot-prompt": """{input}""",
            "propose-prompt": """Question: If there are 5 red apples, 3 green apples, and 2 yellow apples in a basket, and you randomly pick 2 apples, what is the probability that both apples are green?
Possible answers:
A) 1/3
B) 1/6
C) 3/10
D) 1/15
Proposals:
- Total number of apples = 5 red + 3 green + 2 yellow = 10 apples
- dentify the number of favorable outcomes: the number of ways 2 green apples can be selected from 3 is 3
- the total number of ways to choose 2 apples from 10 is 45
- Probability that both apples picked are green 	= \frac{\binom{3}{2}}{\binom{10}{2}}

Question: {question}
Possible answers: {choices}
Proposals:""",
            "value-prompt": """{input}""",
            "vote-prompt": """{input}""",
        },
        "bbh": {
            # CoT 8-shot for evaluation
            "cot-prompt": """{input}""",
            "propose-prompt": """Question: In a race, six runners (X, Y, Z, P, Q, R) finished in the following order: 1st, 2nd, 3rd, 4th, 5th, 6th. Based on the following clues, determine the order in which they finished:
1. P finished ahead of Y.
2. Q finished before Z.
3. R finished immediately before P.
4. X did not finish last.
What is the correct order of the finishers?
Possible answers:
A) X, P, R, Y, Q, Z 
B) X, Q, R, P, Y, Z 
C) X, P, R, Q, Y, Z 
D) X, R, P, Q, Y, Z
Proposals:
- P finished ahead of Y. This means P must be positioned before Y in the final order. *P*Y*
- Q finished before Z. This means Q must be positioned before Z. *Q*Z*
- R finished immediately before P. This means R and P must be adjacent, with R finishing just before P. *RP*
- X did not finish last. This tells us that X cannot be in the 6th position. *X.*

Question: {question}
Possible answers: {choices}
Proposals:""",
            "value-prompt": """{input}""",
            "vote-prompt": """{input}""",
        },
    },
}


def get_prompt(prompt_id: str, dataset_name: str = None, subset_name: str = None):
    # unpack prompt name
    category_name, prompt_name = prompt_id.split(":")
    if subset_name is None:
        dataset_full_name = dataset_name
    else:
        dataset_full_name = f"{dataset_name}-{subset_name}"
    eval_logger.info(f"Loading prompt from {category_name} for {dataset_full_name}")
    if category_name == "promptsource":
        try:
            from promptsource.templates import DatasetTemplates
        except ModuleNotFoundError as exception:
            raise type(exception)(
                "Tried to load a Promptsource template, but promptsource is not installed ",
                "please install promptsource via pip install lm-eval[promptsource] or pip install -e .[promptsource]",
            )
        try:
            if subset_name is None:
                prompts = DatasetTemplates(dataset_name=dataset_name)
            else:
                prompts = DatasetTemplates(
                    dataset_name=dataset_name, subset_name=subset_name
                )
        except Exception:
            raise ValueError(f"{dataset_name} and {subset_name} not found")
        if prompt_name in prompts.all_template_names:
            return prompts[prompt_name]
        else:
            raise ValueError(
                f"{prompt_name} not in prompt list {prompts.all_template_names}"
            )
    elif ".yaml" in category_name:
        import yaml

        with open(category_name, "rb") as file:
            prompt_yaml_file = yaml.full_load(file)

        prompt_string = prompt_yaml_file["prompts"][prompt_name]
        return PromptString(prompt_string)
    else:
        try:
            return PROMPT_REGISTRY[category_name][prompt_name]
        except Exception:
            raise ValueError(
                f"expected only a single `:` as separator between \
                prompt category and name, but got `{prompt_id}` instead"
            )


def load_prompt_list(
    use_prompt: str, dataset_name=None, subset_name=None, yaml_path=None, **kwargs
):
    category_name, prompt_name = use_prompt.split(":")

    if category_name == "promptsource":
        from promptsource.templates import DatasetTemplates

        if subset_name is None:
            prompts = DatasetTemplates(dataset_name=dataset_name)
        else:
            prompts = DatasetTemplates(
                dataset_name=dataset_name, subset_name=subset_name
            )

        prompt_list = utils.pattern_match(prompt_name, prompts.all_template_names)

    elif ".yaml" in category_name:
        import yaml

        if yaml_path is not None:
            category_name = os.path.realpath(os.path.join(yaml_path, category_name))

        with open(category_name, "rb") as file:
            prompt_yaml_file = yaml.full_load(file)

        prompt_list = utils.pattern_match(
            prompt_name, prompt_yaml_file["prompts"].keys()
        )

    # category_name, *prompt_name = use_prompt.split(":")
    # TODO allow to multiple prompt naming
    # if len(prompt_name) > 1:
    #     prompt_list = []
    #     for prompt in prompt_name:
    #         prompt_list.append(utils.pattern_match(prompt_name, prompts.all_template_names))
    # else:
    #     prompt_list = utils.pattern_match(prompt_name, prompts.all_template_names)
    return [":".join([category_name, prompt]) for prompt in prompt_list]


class PromptString:
    def __init__(self, prompt_string):
        self.prompt_string = prompt_string

    def apply(self, doc):
        doc_to_text = self.prompt_string["doc_to_text"]
        doc_to_target = self.prompt_string["doc_to_target"]

        # TODO need a way to process doc_to_choice
        if "doc_to_choice" in self.prompt_string:
            raise NotImplementedError("Not yet implemented to accept doc_to_choice")

        text_string = utils.apply_template(doc_to_text, doc)
        target_string = utils.apply_template(doc_to_target, doc)

        return [text_string, target_string]

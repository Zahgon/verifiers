from typing import List, Tuple

from datasets import Dataset

import verifiers as vf

try:
    import reasoning_gym as rg
    from reasoning_gym.composite import DatasetSpec
    from reasoning_gym.dataset import ProceduralDataset
    from reasoning_gym.utils import SYSTEM_PROMPTS

    DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPTS["default"]
except ImportError as e:
    raise ImportError(
        "ReasoningGymEnv requires reasoning-gym. Install with: uv add 'verifiers[rg]'"
    ) from e


class ReasoningGymEnv(vf.SingleTurnEnv):
    def __init__(
        self,
        gym: str | List[str | dict],
        num_train_examples: int = 1000,
        num_eval_examples: int = 100,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        parser: vf.Parser | None = None,
        seed: int = 0,
    ):
        self.gym = gym
        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples
        self.seed = seed
        total_examples = num_train_examples + num_eval_examples
        self.rg_dataset = self.build_rg_dataset(gym, total_examples, seed=seed)
        dataset, eval_dataset = self.rg_to_hf(self.rg_dataset)
        parser = parser or vf.XMLParser(fields=["answer"])
        rubric = vf.Rubric(parser=parser)

        async def check_answer_reward_func(
            completion: vf.Messages, answer: str, **kwargs
        ) -> float:
            # rg_dataset expects an int index
            pass

        rubric.add_reward_func(check_answer_reward_func)
        rubric.add_reward_func(parser.get_format_reward_func(), weight=0.0)
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            message_type="chat",
        )
        self.parser = parser
        self.rubric = rubric

    def build_rg_dataset(
        self, gym: str | List[str | dict], total_examples: int = 1000, seed: int = 0
    ) -> ProceduralDataset:
        pass

    def rg_to_hf(self, rg_dataset: ProceduralDataset) -> Tuple[Dataset, Dataset]:
        pass

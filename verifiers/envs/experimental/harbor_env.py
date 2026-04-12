import json
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset

import verifiers as vf
from verifiers.utils.import_utils import load_toml

logger = logging.getLogger(__name__)


class HarborEnv(vf.CliAgentEnv):
    """CliAgentEnv subclass that loads Harbor-format tasks."""

    def __init__(
        self,
        run_command: str,
        dataset_path: str | Path,
        tasks: list[str] | None = None,
        agent_workdir: str = "/app",
        docker_image: str = "python:3.11-slim",
        **kwargs,
    ):
        self.dataset_path = Path(dataset_path)
        self.task_names = tasks
        self.agent_workdir = agent_workdir

        kwargs["docker_image"] = docker_image

        dataset = self.load_harbor_dataset()
        rubric = vf.Rubric(funcs=[self.harbor_reward], weights=[1.0])

        super().__init__(
            run_command=run_command, dataset=dataset, rubric=rubric, **kwargs
        )

    def load_harbor_dataset(self) -> Dataset:
        """Load Harbor tasks from dataset directory into a Dataset with prompts."""
        pass

    async def get_docker_image(self, state: vf.State) -> str:
        """Get Docker image from task info, falling back to default."""
        task_info: dict[str, Any] = state.get("info") or {}
        return task_info.get("docker_image") or self.docker_image

    async def build_env_vars(self, state: vf.State) -> dict[str, str]:
        """Build env vars with Harbor-specific additions."""
        env_vars = await super().build_env_vars(state)
        env_vars.setdefault("HARBOR_TASK_NAME", state.get("task", ""))
        env_vars.setdefault("HARBOR_TASK_DIR", "/task")
        env_vars.setdefault("HARBOR_INSTRUCTION_PATH", "/task/instruction.md")
        if self.agent_workdir:
            env_vars.setdefault("AGENT_WORKDIR", self.agent_workdir)
        return env_vars

    async def post_sandbox_setup(self, state: vf.State) -> None:
        """Upload Harbor task assets after sandbox creation."""
        task_info: dict[str, Any] = state.get("info", {}) or {}
        task_dir_str = task_info.get("task_dir", "")
        if not task_dir_str:
            raise ValueError("task_dir not set in task info")
        task_dir = Path(task_dir_str)
        config = task_info.get("config", {})

        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        await self.prepare_harbor_task(state["sandbox_id"], task_dir)
        state["harbor_config"] = config
        state["harbor_task_dir"] = str(task_dir)

    async def prepare_harbor_task(self, sandbox_id: str, task_dir: Path) -> None:
        """Upload task instruction only (oracle/tests uploaded after agent completes)."""
        instruction_path = task_dir / "instruction.md"
        task_toml_path = task_dir / "task.toml"

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                if instruction_path.exists():
                    tar.add(instruction_path, arcname="task/instruction.md")

                if task_toml_path.exists():
                    tar.add(task_toml_path, arcname="task/task.toml")

            remote_tar = "/tmp/harbor_task.tar.gz"
            await self.sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))
            await self.sandbox_client.execute_command(
                sandbox_id,
                f"mkdir -p /task /logs/verifier {self.agent_workdir} && "
                f"tar -xzf {remote_tar} -C / && rm {remote_tar}",
                working_dir=None,
            )
            logger.debug(f"Uploaded task instruction for {task_dir.name}")
        finally:
            tar_path.unlink(missing_ok=True)

    async def upload_test_assets(self, sandbox_id: str, task_dir: Path) -> None:
        """Upload oracle/tests after agent completes, right before running tests."""
        pass

    async def post_rollout(self, state: vf.State):
        """Run Harbor tests to compute reward before sandbox destruction."""
        pass

    async def harbor_reward(self, state: vf.State, **kwargs) -> float:
        pass

    async def compute_reward(self, state: vf.State) -> float:
        """
        Execute Harbor tests (tests/test.sh) inside the sandbox to compute reward.
        Uploads oracle/tests first (they don't exist during agent execution).
        Prioritizes /logs/verifier/reward.txt, falling back to reward.json.
        """
        pass

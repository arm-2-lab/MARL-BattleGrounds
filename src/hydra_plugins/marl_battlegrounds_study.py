"""Send finite Hydra sweeps to MARL-BGs' existing study runner.

The optional hydra extra provides this namespace plugin. It composes all jobs
before dispatch; it does not own worker scheduling, training or recovery.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn, JobStatus
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf, open_dict


class StudyLauncher(Launcher):
    """Compose one finite batch and return each saved study case's outcome."""

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """Bind Hydra's composer and reject unsupported execution owners early.

        task_function is intentionally unused: the study calls the public trainer.
        Only BasicSweeper with max_batch_size=null is supported. Worker environment
        belongs to resources/inherited settings; per-job Hydra environment changes,
        directory changes and extra callbacks are refused rather than ignored.
        """
        self._check_controls(config)
        self.config = config
        self.context = hydra_context
        self.used = False

    @staticmethod
    def _check_controls(config: DictConfig) -> None:
        """Reject unsupported scheduler or worker controls in every composed job."""
        if (
            config.hydra.launcher._target_
            != "hydra_plugins.marl_battlegrounds_study.StudyLauncher"
        ):
            raise ValueError(
                "Nested training launchers are unsupported; use the study launcher"
            )
        if (
            config.hydra.sweeper._target_
            != "hydra._internal.core_plugins.basic_sweeper.BasicSweeper"
            or config.hydra.sweeper.max_batch_size is not None
        ):
            raise ValueError(
                "Use Hydra BasicSweeper with max_batch_size=null for one finite study"
            )
        if (
            config.hydra.job.chdir
            or config.hydra.job.env_set
            or config.hydra.job.env_copy
        ):
            raise ValueError(
                "Study workers use resources and the inherited environment; "
                "Hydra chdir/env_set/env_copy are unsupported"
            )
        if set(config.hydra.callbacks) - {"launch"}:
            raise ValueError(
                "Study sweeps do not run Hydra job callbacks; "
                "use public training callbacks"
            )

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        """Compose the complete ordered batch, then call the one study owner.

        Reject repeated/partial batches. Resolve each task with its own Hydra
        context, restoring the caller's global state even on failure. Save reports
        before returning failed JobReturns; Hydra may raise when it reads them.
        Preview returns clearly labelled software previews, never trained models.
        """
        from marl_battlegrounds.experiments.hydra import _plain_settings, _study_sweep

        if self.used or initial_job_idx != 0 or not job_overrides:
            raise ValueError(
                "Study launcher requires one complete nonempty sweep batch"
            )
        self.used = True
        original = HydraConfig.instance().cfg
        jobs: list[dict[str, Any]] = []
        task_configs: list[DictConfig] = []
        hydra_configs: list[DictConfig] = []
        try:
            for index, overrides in enumerate(job_overrides):
                composed = self.context.config_loader.load_sweep_config(
                    self.config, list(overrides)
                )
                self._check_controls(composed)
                with open_dict(composed):
                    composed.hydra.job.id = str(index)
                    composed.hydra.job.num = index
                HydraConfig.instance().set_config(composed)
                task = OmegaConf.masked_copy(
                    composed, [str(key) for key in composed if key != "hydra"]
                )
                jobs.append(_plain_settings(task))
                task_configs.append(OmegaConf.create(jobs[-1]))
                hydra_configs.append(OmegaConf.masked_copy(composed, "hydra"))
        finally:
            HydraConfig.instance().cfg = original
        base = Path(self.config.hydra.runtime.cwd)
        report = _study_sweep(
            jobs,
            overrides=[list(row) for row in job_overrides],
            base_dir=base,
            launch_dir=Path(self.config.hydra.sweep.dir),
            sweep_overrides=cast(
                list[str],
                OmegaConf.to_container(self.config.hydra.overrides.task, resolve=False),
            ),
            sweeper_params={}
            if self.config.hydra.sweeper.params is None
            else OmegaConf.to_container(
                self.config.hydra.sweeper.params, resolve=False
            ),
        )
        results: list[JobReturn] = []
        for index, row in enumerate(report["cases"]):
            complete = report.get("preview", False) or row["status"] == "complete"
            result = JobReturn(
                overrides=list(job_overrides[index]),
                cfg=task_configs[index],
                hydra_cfg=hydra_configs[index],
                working_dir=str(base),
                task_name=str(self.config.hydra.job.name),
                status=JobStatus.COMPLETED if complete else JobStatus.FAILED,
            )
            result.return_value = (
                row
                if complete
                else RuntimeError(
                    f"Study case {row['id']} is {row['status']}; "
                    f"saved report: {report['report_path']}"
                )
            )
            results.append(result)
        return results

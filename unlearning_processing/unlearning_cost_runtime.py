#!/usr/bin/env python3
"""Shared runtime for step-accurate unlearning cost measurements."""

import gc
import json
import math
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch


class CostMeasurementStop(RuntimeError):
    """Controlled termination after success or exhaustion of the step budget."""


class CostMeasurementFailed(RuntimeError):
    """Raised after a cost run has persisted a failure result."""


def read_cost_config(config_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not config_path:
        return None
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def cost_training_log_dir(config_path: Optional[str], fallback: str) -> str:
    config = read_cost_config(config_path)
    if config is None:
        return fallback
    return str(config["training_log_dir"])


def _atomic_json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    os.replace(str(temporary), str(path))


class CostMeasurementController:
    """Times gradient work and synchronously evaluates selected checkpoints."""

    def __init__(
        self,
        config: Dict[str, Any],
        dataset_required: Any,
        dataset_remained: Any,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.dataset_required = dataset_required
        self.dataset_remained = dataset_remained
        self.clock = clock
        self.max_steps = int(config["maximum_steps"])
        self.interval = int(config["eval_interval"])
        threshold_config = config.get("threshold") or {}
        threshold_value = threshold_config.get("value")
        self.threshold = (
            float(threshold_value) if threshold_value is not None else None
        )
        self.trajdeleter_model_dir = config.get("trajdeleter_model_dir")
        self._reference_ready = self.trajdeleter_model_dir is None
        self.gpu = int(config.get("gpu", 0))
        self.batch_size = int(config.get("batch_size", 512))
        self.forget_sample_episodes = int(
            config.get("forget_sample_episodes", 100)
        )
        self.result_path = Path(config["job_result_path"])
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training_log_dir = str(config["training_log_dir"])
        Path(self.training_log_dir).mkdir(parents=True, exist_ok=True)

        if self.forget_sample_episodes <= 0:
            raise ValueError("forget_sample_episodes must be positive.")
        if self.max_steps <= 0 or self.interval <= 0:
            raise ValueError("maximum_steps and eval_interval must be positive.")
        if self.threshold is None and self.trajdeleter_model_dir is None:
            raise ValueError(
                "Either a threshold value or trajdeleter_model_dir is required."
            )
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("The TrajDeleter threshold must be finite.")

        self.steps_trained = 0
        self.pass_streak = 0
        self.training_wall_seconds = 0.0
        self.evaluations = []
        self.checkpoints = []
        self._active_since: Optional[float] = None
        self._states_f = None
        self._finished = False

        source_params = Path(config["original_model_dir"]) / "params.json"
        if source_params.exists():
            shutil.copy2(str(source_params), str(self.checkpoint_dir / "params.json"))

        self._write_result("running")

    @classmethod
    def from_path(
        cls,
        config_path: Optional[str],
        dataset_required: Any,
        dataset_remained: Any,
    ) -> Optional["CostMeasurementController"]:
        config = read_cost_config(config_path)
        if config is None:
            return None
        return cls(config, dataset_required, dataset_remained)

    def start_training(self) -> None:
        # Callers must invoke this only after setup/precomputation is complete.
        # v21 builds the fixed original-reference cache before this boundary.
        if self._active_since is None and not self._finished:
            self._sync_cuda()
            self._active_since = self.clock()

    def _sync_cuda(self) -> None:
        if self.gpu >= 0 and torch.cuda.is_available():
            torch.cuda.synchronize(self.gpu)

    def _pause_training_clock(self) -> None:
        if self._active_since is None:
            return
        self._sync_cuda()
        self.training_wall_seconds += self.clock() - self._active_since
        self._active_since = None

    def _resume_training_clock(self) -> None:
        if not self._finished:
            self._sync_cuda()
            self._active_since = self.clock()

    def should_evaluate(self, step: int) -> bool:
        return step > 0 and (step % self.interval == 0 or step == self.max_steps)

    def after_update(
        self,
        algorithm: Any,
        logical_step: Optional[int] = None,
        original_model: Optional[Any] = None,
    ) -> None:
        if self._finished:
            raise CostMeasurementStop("Cost measurement already finished.")
        if self._active_since is None:
            self.start_training()

        if logical_step is None:
            self.steps_trained += 1
        else:
            if logical_step <= self.steps_trained:
                raise ValueError(
                    "Cost measurement logical steps must be strictly increasing: "
                    f"previous={self.steps_trained}, current={logical_step}"
                )
            self.steps_trained = int(logical_step)

        if self.steps_trained > self.max_steps:
            self._finish("not_achieved", None, self.max_steps)
        if not self.should_evaluate(self.steps_trained):
            return

        self._pause_training_clock()
        checkpoint_path = self.checkpoint_dir / f"model_{self.steps_trained}.pt"
        try:
            algorithm.save_model(str(checkpoint_path))
            self.checkpoints.append(checkpoint_path)
            evaluation_started = self.clock()
            statistics = self._evaluate(algorithm, original_model)
            evaluation_wall_seconds = self.clock() - evaluation_started
            metric = float(statistics["D_f_W_unlearn_retrain"])
            if not math.isfinite(metric):
                raise ValueError("D_f_W_unlearn_retrain is not finite.")
            if self.threshold is None:
                raise ValueError("TrajDeleter reference threshold is not ready.")

            passed = metric <= self.threshold
            self.pass_streak = self.pass_streak + 1 if passed else 0
            self.evaluations.append(
                {
                    "step": self.steps_trained,
                    "D_f_W_unlearn_retrain": metric,
                    "threshold": self.threshold,
                    "passed": passed,
                    "consecutive_passes": self.pass_streak,
                    "evaluation_wall_seconds": evaluation_wall_seconds,
                    "checkpoint": str(checkpoint_path),
                }
            )
            self._write_result("running")

            if self.pass_streak >= 2:
                self._finish("achieved", self.steps_trained, self.steps_trained)
            if self.steps_trained >= self.max_steps:
                self._finish("not_achieved", None, self.max_steps)
        except CostMeasurementStop:
            raise
        except Exception as error:
            self.fail(error)
        finally:
            if not self._finished:
                self._restore_training_mode(algorithm)
                # The next update wrapper restarts the clock immediately before
                # gradient work. Keeping it paused here prevents checkpoint,
                # evaluator, logger, and batch-loader overhead from leaking into
                # the measured interval after an evaluation.

    def _prepare_states(self) -> None:
        if self._states_f is not None:
            return
        from Offline_RL_processing.evaluate_critic_orl_like import (
            extract_sampled_forget_states,
        )

        self._states_f = extract_sampled_forget_states(
            self.dataset_remained,
            sample_episodes=self.forget_sample_episodes,
            random_seed=int(self.config.get("seed", 0)),
        )

    def _evaluate(self, algorithm: Any, original_model: Optional[Any]) -> Dict[str, float]:
        from Offline_RL_processing.evaluate_critic_orl_like import (
            evaluate_critic_orl_like_forget_only,
            load_model_from_dir,
        )

        self._prepare_states()
        loaded_original = None
        retrain_model = None
        trajdeleter_model = None
        try:
            if original_model is None:
                loaded_original, _ = load_model_from_dir(
                    self.config["original_model_dir"], self.gpu
                )
                original_for_eval = loaded_original
            else:
                original_for_eval = original_model
            retrain_model, _ = load_model_from_dir(
                self.config["retrain_model_dir"], self.gpu
            )
            if not self._reference_ready:
                trajdeleter_model, _ = load_model_from_dir(
                    self.trajdeleter_model_dir, self.gpu
                )
                reference_statistics = evaluate_critic_orl_like_forget_only(
                    original_for_eval,
                    retrain_model,
                    trajdeleter_model,
                    self._states_f,
                    batch_size=self.batch_size,
                )
                reference_metric = float(
                    reference_statistics["D_f_W_unlearn_retrain"]
                )
                if not math.isfinite(reference_metric):
                    raise ValueError(
                        "TrajDeleter reference D_f_W_unlearn_retrain "
                        "is not finite."
                    )
                self.threshold = reference_metric
                threshold_config = dict(self.config.get("threshold") or {})
                threshold_config.update(
                    {
                        "value": reference_metric,
                        "metric": "D_f_W_unlearn_retrain",
                        "source": "same_sample_trajdeleter_model",
                        "source_model_dir": str(self.trajdeleter_model_dir),
                    }
                )
                self.config["threshold"] = threshold_config
                self._reference_ready = True

            return evaluate_critic_orl_like_forget_only(
                original_for_eval,
                retrain_model,
                algorithm,
                self._states_f,
                batch_size=self.batch_size,
            )
        finally:
            if loaded_original is not None:
                del loaded_original
            if retrain_model is not None:
                del retrain_model
            if trajdeleter_model is not None:
                del trajdeleter_model
            gc.collect()
            if self.gpu >= 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _restore_training_mode(algorithm: Any) -> None:
        try:
            from d3rlpy.torch_utility import set_train_mode

            if getattr(algorithm, "impl", None) is not None:
                set_train_mode(algorithm.impl)
        except Exception:
            pass

    def _base_result(self, status: str) -> Dict[str, Any]:
        final_metric = (
            self.evaluations[-1]["D_f_W_unlearn_retrain"]
            if self.evaluations else None
        )
        return {
            "schema_version": 1,
            "method": self.config["method"],
            "seed": int(self.config["seed"]),
            "task": self.config["task"],
            "threshold": self.config["threshold"],
            "original_model_dir": self.config["original_model_dir"],
            "retrain_model_dir": self.config["retrain_model_dir"],
            "command": self.config.get("command"),
            "status": status,
            "steps_to_threshold": None,
            "steps_trained": int(self.steps_trained),
            "training_wall_seconds": float(self.training_wall_seconds),
            "gpu_hours": (
                float(self.training_wall_seconds / 3600.0)
                if self.gpu >= 0 else 0.0
            ),
            "final_D_f_W_unlearn_retrain": final_metric,
            "evaluations": self.evaluations,
            "retained_checkpoint": None,
            "error": None,
        }

    def _write_result(self, status: str, **updates: Any) -> None:
        payload = self._base_result(status)
        payload.update(updates)
        _atomic_json_dump(self.result_path, payload)

    def _finish(
        self,
        status: str,
        steps_to_threshold: Optional[int],
        steps_trained: int,
    ) -> None:
        self._pause_training_clock()
        self._finished = True
        self.steps_trained = int(steps_trained)
        retained = self.checkpoints[-1] if self.checkpoints else None
        self._prune_checkpoints(retained)
        self._write_result(
            status,
            steps_to_threshold=steps_to_threshold,
            steps_trained=self.steps_trained,
            retained_checkpoint=str(retained) if retained is not None else None,
        )
        raise CostMeasurementStop(status)

    def _prune_checkpoints(self, retained: Optional[Path]) -> None:
        for checkpoint in self.checkpoints:
            if retained is not None and checkpoint == retained:
                continue
            try:
                checkpoint.unlink()
            except FileNotFoundError:
                pass

    def fail(self, error: BaseException) -> None:
        self._pause_training_clock()
        self._finished = True
        retained = self.checkpoints[-1] if self.checkpoints else None
        self._prune_checkpoints(retained)
        self._write_result(
            "failed",
            retained_checkpoint=str(retained) if retained is not None else None,
            error="".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        )
        raise CostMeasurementFailed(str(error)) from error

    def finish_if_incomplete(self) -> None:
        if self._finished:
            return
        self.fail(
            RuntimeError(
                f"Training exited after {self.steps_trained} logical steps; "
                f"expected the controller to stop it at maximum_steps={self.max_steps}."
            )
        )


def wrap_negative_reward_updates(
    algorithm: Any,
    controller: CostMeasurementController,
    original_model: Optional[Any] = None,
) -> None:
    """Count both retain and negative-reward updates as logical gradient steps."""
    original_remain_update = algorithm.update_stage1_remain
    original_unlearn_update = algorithm.update_stage1_unlearn

    def wrapped_remain_update(batch: Any):
        controller.start_training()
        result = original_remain_update(batch)
        controller.after_update(algorithm, original_model=original_model)
        return result

    def wrapped_unlearn_update(batch: Any, alpha: float):
        controller.start_training()
        result = original_unlearn_update(batch, alpha)
        controller.after_update(algorithm, original_model=original_model)
        return result

    algorithm.update_stage1_remain = wrapped_remain_update
    algorithm.update_stage1_unlearn = wrapped_unlearn_update


def wrap_standard_updates(
    algorithm: Any,
    controller: CostMeasurementController,
    original_model: Optional[Any] = None,
) -> None:
    """Instrument d3rlpy's ordinary update entry point without timing fit setup."""
    original_update = algorithm.update

    def wrapped_update(batch: Any):
        controller.start_training()
        result = original_update(batch)
        controller.after_update(algorithm, original_model=original_model)
        return result

    algorithm.update = wrapped_update
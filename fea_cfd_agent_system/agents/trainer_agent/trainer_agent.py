"""
Trainer Agent — auto-configures and trains the selected model.
Applies physics-informed loss from Physics Agent feedback.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from agents.orchestrator.agent_state import (
    AgentSystemState, AgentStatus, TrainingResult
)
from agents.trainer_agent.physics_loss import PhysicsInformedLoss
from agents.trainer_agent.auto_configurator import AutoConfigurator


class TrainerAgent:

    def __init__(self, config: dict):
        self.config = config
        self.configurator = AutoConfigurator(config)

    def run(self, state: AgentSystemState) -> AgentSystemState:
        model_name = state.selected_model.name if state.selected_model else "unknown"
        logger.info(f"Trainer Agent: training {model_name}")
        state.trainer_status = AgentStatus.RUNNING

        try:
            hints = ""
            if state.selected_model and state.selected_model.github_report:
                hints = state.selected_model.github_report.get("config_hints", "")

            # Check if custom model object is already built
            custom_model_obj = None
            if (state.selected_model and
                    state.selected_model.github_report.get("custom_model") and
                    state.selected_model.github_report.get("model_object") is not None):
                custom_model_obj = state.selected_model.github_report["model_object"]

            model, config_used = self.configurator.build_model(
                model_name=model_name,
                problem_card=state.problem_card,
                dataset=state.dataset,
                config_hints=hints,
                use_optuna=state.unified_schema.get("use_optuna", False),
                custom_model=custom_model_obj,
            )

            loss_fn = PhysicsInformedLoss(
                lambda_weights=state.physics_lambda_weights,
                physics_type=state.problem_card.physics_type.value,
            )

            result = self._train(model, state.dataset, loss_fn, config_used)
            state.training_result = result
            state.trainer_status = AgentStatus.PASSED

            logger.success(
                f"Training complete: loss={result.val_loss:.6f}, "
                f"epochs={result.training_epochs}, "
                f"converged={result.converged}"
            )

        except Exception as e:
            state.trainer_status = AgentStatus.FAILED
            state.training_result = TrainingResult()
            state.error_message = f"Trainer failed: {e}"
            logger.error(state.error_message)

        return state

    def _train(self, model, dataset, loss_fn, config) -> TrainingResult:
        """Core training loop with early stopping."""
        max_epochs = config.get("max_epochs", self.config.get("training", {}).get("max_epochs", 500))
        patience   = config.get("patience",   self.config.get("training", {}).get("early_stopping_patience", 20))
        lr         = config.get("lr",         self.config.get("training", {}).get("default_lr", 1e-3))
        clip_norm  = self.config.get("training", {}).get("gradient_clip_max_norm", 1.0)

        cases     = dataset.get("cases", [])
        n_val     = max(1, int(len(cases) * 0.2))
        train_cases = cases[:-n_val]
        val_cases   = cases[-n_val:]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_val_loss    = float("inf")
        patience_counter = 0
        train_loss       = 0.0
        start_time       = time.time()
        epoch            = 0

        for epoch in range(max_epochs):
            model.train()
            train_loss = 0.0
            for case in train_cases:
                x, y = self._case_to_tensors(case)
                if x is None:
                    continue
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y, case)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_cases), 1)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for case in val_cases:
                    x, y = self._case_to_tensors(case)
                    if x is None:
                        continue
                    pred = model(x)
                    loss = loss_fn(pred, y, case)
                    val_loss += loss.item()
            val_loss /= max(len(val_cases), 1)

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 100 == 0:
                logger.debug(f"  Epoch {epoch + 1}: train={train_loss:.6f}, val={val_loss:.6f}")

        return TrainingResult(
            model_object=model,
            train_loss=train_loss,
            val_loss=best_val_loss,
            training_epochs=epoch + 1,
            training_time_seconds=time.time() - start_time,
            converged=patience_counter < patience,
            config_used=config,
        )

    def _case_to_tensors(self, case):
        """Convert a simulation case to input/output tensors."""
        coords = case.get("node_coords")
        fields = case.get("fields", {})
        if coords is None or not fields:
            return None, None
        x = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
        targets = [f for f in fields.values() if isinstance(f, np.ndarray)]
        if not targets:
            return None, None
        y = torch.tensor(np.concatenate([t.flatten() for t in targets]), dtype=torch.float32)
        return x, y

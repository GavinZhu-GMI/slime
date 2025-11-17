import os
import socket
import time
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.ray.train_actor import TrainRayActor
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group, init_process_group
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.ray_utils import Box
from slime.utils.timer import Timer, timer
from slime.utils.wandb_utils import init_wandb_secondary

from .checkpoint import load_checkpoint
from .cp_utils import slice_log_prob_with_cp
from .data import DataIterator, get_data_iterator, log_perf_data, log_rollout_data, sync_actor_critic_data
from .initialize import init, is_megatron_main_rank
from .loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor, named_parameters


class MegatronTrainRayActor(TrainRayActor):
    def init(
        self,
        args: Namespace,
        role: str,
        wandb_run_id: str,
        with_ref: bool = False,
    ) -> Optional[int]:
        super().init(args, role, wandb_run_id, with_ref)

        init(args)

        if is_megatron_main_rank():
            init_wandb_secondary(args, wandb_run_id)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        if self.args.debug_rollout_only:
            Timer().start("train_wait")
            return 0

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr

        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args, role
        )

        if role == "critic":
            if self.args.offload:
                self.sleep(("model"))
            Timer().start("train_wait")
            return

        start_rollout_id = loaded_rollout_id + 1
        self.weights = {"actor": {}}
        self.update_cpu_params_dict(self.weights["actor"])

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights["rollout_actor"] = {}
                self.update_cpu_params_dict(self.weights["rollout_actor"])

        update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            self.weights,
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            vocab_size=self.tokenizer.vocab_size if self.args.vocab_size is None else self.args.vocab_size,
        )

        # empty cache after initialization
        clear_memory()

        if self.args.offload:
            # recover to actor in the end.
            self.update_gpu_params_dict(self.weights["actor"])
            self.sleep(("model"))

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof = None
        if args.use_pytorch_profiler and torch.distributed.get_rank() == 0:
            self.prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=max(args.profile_step_start - 1, 0),
                    warmup=1 if args.profile_step_start > 0 else 0,
                    active=args.profile_step_end - args.profile_step_start,
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                with_flops=True,
            )
            self.prof.start()

        Timer().start("train_wait")
        return start_rollout_id

    @torch.no_grad()
    def update_cpu_params_dict(self, params_dict: Dict[str, torch.Tensor]) -> None:
        for name, param in named_parameters(self.args, self.model):
            if name not in params_dict:
                params_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            params_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def update_gpu_params_dict(self, params_dict: Dict[str, torch.Tensor]) -> None:
        for name, param in named_parameters(self.args, self.model):
            assert name in params_dict
            param.copy_(params_dict[name], non_blocking=True)
        torch.cuda.synchronize()

    @timer
    def sleep(self, tags: Union[str, Tuple[str, ...]]) -> None:
        assert self.args.offload
        assert "model" in tags
        if isinstance(tags, str):
            tags = (tags,)

        clear_memory()
        print_memory("before offload model")
        if hasattr(mpu, "destroy_process_groups"):
            mpu.destroy_process_groups()

        torch_memory_saver.pause()

        print_memory("after offload model")

    @timer
    def wake_up(self, tags: Union[str, Tuple[str, ...]]) -> None:
        assert self.args.offload

        # there are weird times when sglang is not offloaded immediately, so we wait here.
        mem_fraction_static = self.args.sglang_mem_fraction_static or 0.8
        for _ in range(60):
            memory_info = print_memory("before wake_up model")
            if memory_info["used_GB"] >= mem_fraction_static * memory_info["total_GB"]:
                time.sleep(1)
                continue
            break

        if isinstance(tags, str):
            tags = (tags,)

        torch_memory_saver.resume()

        clear_memory()
        if hasattr(mpu, "reload_process_groups"):
            mpu.reload_process_groups()
        print_memory("after wake_up model")

    def _get_rollout_data(
        self, rollout_data_ref: Box
    ) -> Dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]:
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will recieve the data.
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        # TODO: this is ugly, move to somewhere else?
        # move tokens to GPU in advance
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in rollout_data["loss_masks"]
        ]
        if "rollout_log_probs" in rollout_data:
            rollout_data["rollout_log_probs"] = [
                torch.tensor(
                    slice_log_prob_with_cp(log_prob, total_length, response_length),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for log_prob, total_length, response_length in zip(
                    rollout_data["rollout_log_probs"], rollout_data["total_lengths"], rollout_data["response_lengths"]
                )
            ]

        # Move RL fields to GPU if they exist and are not already on GPU
        # This is needed for external APIs (like GMI wrapper) that provide pre-computed RL fields
        # In normal Slime training, these fields come from GPU forward passes and don't need moving
        if getattr(self.args, 'move_rl_fields_to_gpu', False):
            for field in ["log_probs", "ref_log_probs", "advantages", "returns", "values"]:
                if field in rollout_data and rollout_data[field]:
                    # Check if first tensor is already on GPU to avoid unnecessary transfers
                    first_tensor = rollout_data[field][0]
                    if isinstance(first_tensor, torch.Tensor) and not first_tensor.is_cuda:
                        rollout_data[field] = [
                            torch.tensor(t, dtype=torch.float32, device=torch.cuda.current_device())
                            if not isinstance(t, torch.Tensor) or not t.is_cuda
                            else t.to(device=torch.cuda.current_device())
                            for t in rollout_data[field]
                        ]

        return rollout_data

    def compute_log_prob(
        self,
        model_tag: str,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> Dict[str, list[torch.Tensor]]:
        self.update_gpu_params_dict(self.weights[model_tag])

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        Timer().end("train_wait")

        if self.args.offload:
            self.wake_up(("model"))

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)
            if self.args.debug_rollout_only:
                log_rollout_data(rollout_id, self.args, rollout_data)
                Timer().start("train_wait")
                return

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def train_critic(
        self, rollout_id: int, rollout_data: Dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]
    ) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
        )
        Timer().start("train_wait")

    def train_actor(
        self, rollout_id: int, rollout_data: Dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]
    ) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        with timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    rollout_data.update(
                        self.compute_log_prob(
                            "ref",
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )

                if self.args.use_routing_replay:
                    os.environ["ROUTING_REPLAY_STAGE"] = "record"
                rollout_data.update(
                    self.compute_log_prob(
                        "old_actor" if self.args.keep_old_actor else "actor",
                        data_iterator,
                        num_microbatches,
                        store_prefix="",
                    )
                )

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )

                # when there is old actor, we need to update the model params to actor manually
                if "old_actor" in self.weights:
                    self.update_gpu_params_dict(self.weights["actor"])

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            if self.args.use_pytorch_profiler and torch.distributed.get_rank() == 0 and self.prof is not None:
                self.prof.step()

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            # Profiling.
            if (
                self.args.use_pytorch_profiler
                and rollout_id == self.args.profile_step_end
                and torch.distributed.get_rank() == 0
                and self.prof is not None
            ):
                self.prof.stop()
                self.prof = None

        # TODO extract to a function during refactor
        if (path_template := self.args.save_debug_train_data) is not None:
            rank = torch.distributed.get_rank()
            path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
            print(f"Save debug train data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(
                    rollout_id=rollout_id,
                    rank=rank,
                    rollout_data=rollout_data,
                ),
                path,
            )

        if self.args.use_routing_replay:
            from megatron.core.transformer.moe.moe_utils import RoutingReplay

            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        self.update_cpu_params_dict(self.weights["actor"])

        log_perf_data(rollout_id, self.args)
        Timer().start("train_wait")

    def forward_backward_step_only(
        self, rollout_id: int, rollout_data_ref: Box, zero_grads: bool = False
    ) -> Dict[str, float]:
        """
        Perform forward + backward pass only, accumulating gradients WITHOUT optimizer.step().

        This enables gradient accumulation by calling Megatron's forward_backward_func directly
        without the optimizer step.

        Args:
            rollout_id: Rollout identifier
            rollout_data_ref: Reference to rollout data
            zero_grads: If True, zero gradients before forward pass (first accumulation step).
                       If False, accumulate on top of existing gradients (subsequent steps).

        Returns:
            Dictionary with loss, grad_norm, and valid_step information
        """
        import math
        import os
        from functools import partial

        from megatron.core import mpu
        from megatron.core.models.gpt import GPTModel
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.training.global_vars import get_args

        from .data import get_batch
        from .loss import loss_function

        Timer().end("train_wait")

        if self.args.offload:
            self.wake_up(("model"))

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        # Create data iterator
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        with timer("forward_backward_only"):
            args = get_args()

            # Optionally zero gradients (only for first accumulation step)
            if zero_grads:
                for model_chunk in self.model:
                    model_chunk.zero_grad_buffer()
                self.optimizer.zero_grad()

            if args.custom_megatron_before_train_step_hook_path:
                from slime.utils.misc import load_function

                custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
                custom_before_train_step_hook(args, rollout_id, 0, self.model, self.optimizer, self.opt_param_scheduler)

            def forward_step(data_iterator, model: GPTModel):
                """Forward training step."""
                batch = get_batch(
                    data_iterator,
                    [
                        "tokens",
                        "packed_seq_params",
                        "total_lengths",
                        "response_lengths",
                        "loss_masks",
                        "log_probs",
                        "ref_log_probs",
                        "values",
                        "advantages",
                        "returns",
                        "rollout_log_probs",
                    ],
                )

                if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
                    old_stage = os.environ["ROUTING_REPLAY_STAGE"]
                    os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

                output_tensor = model(
                    input_ids=batch["tokens"],
                    position_ids=None,
                    attention_mask=None,
                    labels=None,
                    packed_seq_params=batch["packed_seq_params"],
                )

                if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
                    os.environ["ROUTING_REPLAY_STAGE"] = old_stage

                return output_tensor, partial(loss_function, args, batch, num_microbatches[0])

            # Call Megatron's forward_backward_func directly
            forward_backward_func = get_forward_backward_func()
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=data_iterator,
                model=self.model,
                num_microbatches=num_microbatches[0],
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=False,
            )

            # Validate gradients
            valid_step = True
            grad_norm = None
            if not getattr(args, "check_for_nan_in_loss_and_grad", True):
                found_inf_flag = self.optimizer.prepare_grads()
                if found_inf_flag:
                    valid_step = False
                else:
                    grad_norm = self.optimizer.get_grad_norm()
                    if isinstance(grad_norm, torch.Tensor):
                        valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
                    else:
                        valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

            # Compute loss metrics
            loss_dict = {}
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                keys = losses_reduced[0]["keys"]
                values = None
                for x in losses_reduced:
                    if values is None:
                        values = x["values"]
                    else:
                        values += x["values"]
                assert len(keys) + 1 == values.numel()
                torch.distributed.all_reduce(values, group=mpu.get_data_parallel_group(with_context_parallel=True))

                values = values.tolist()
                num_samples_or_tokens = values[0]
                for key, value in zip(keys, values[1:]):
                    loss_dict[key] = value * mpu.get_context_parallel_world_size() / num_samples_or_tokens

                # Extract log_probs if present (added separately, not in keys/values tensor)
                # Aggregate log_probs from all microbatches
                if "log_probs" in losses_reduced[0]:
                    # log_probs is a list of tensors (one per sample in batch)
                    # With multiple microbatches, we need to concatenate across microbatches
                    all_log_probs = []
                    for x in losses_reduced:
                        if "log_probs" in x and x["log_probs"]:
                            all_log_probs.extend(x["log_probs"])
                    loss_dict["log_probs"] = all_log_probs

        Timer().start("train_wait")

        return {
            "loss": loss_dict,
            "grad_norm": grad_norm if grad_norm is not None else 0.0,
            "valid_step": valid_step,
        }

    def forward_only_step(
        self, rollout_id: int, rollout_data_ref: Box
    ) -> Dict[str, Any]:
        """
        Perform forward-only pass WITHOUT gradients, returning logprobs per sample.

        This is used for DPO's forward_backward_custom where we need reference logprobs
        from a forward pass, then apply custom loss function client-side.

        Unlike forward_backward_step_only, this does NOT compute gradients.

        Args:
            rollout_id: Rollout identifier
            rollout_data_ref: Reference to rollout data

        Returns:
            Dictionary with loss_dict containing log_probs per sample
        """
        from megatron.core import mpu

        from .loss import get_log_probs_and_entropy
        from .model import forward_only

        Timer().end("train_wait")

        if self.args.offload:
            self.wake_up(("model"))

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        # Create data iterator
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        with timer("forward_only"):
            # Call forward_only which sets model to eval mode and does forward pass only
            rollout_data_result = forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix="",
            )

            # Extract log_probs from rollout_data_result
            # forward_only returns dict with "log_probs" key containing list of tensors
            loss_dict = {}
            if mpu.is_pipeline_last_stage():
                if "log_probs" in rollout_data_result:
                    loss_dict["log_probs"] = rollout_data_result["log_probs"]
                if "entropy" in rollout_data_result:
                    loss_dict["entropy"] = rollout_data_result["entropy"]

        Timer().start("train_wait")

        return {
            "loss": loss_dict,
            "grad_norm": 0.0,  # No gradients computed in forward-only pass
            "valid_step": True,  # Always valid since no gradient computation
        }

    def apply_optimizer_step(self) -> Dict[str, float]:
        """
        Apply optimizer step using accumulated gradients.

        This enables gradient accumulation by applying the optimizer step
        and zeroing gradients after.

        Returns:
            Dictionary with success status and grad_norm
        """
        from megatron.training.global_vars import get_args

        with timer("apply_optimizer_step"):
            args = get_args()

            # Apply optimizer step
            update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

            # Update learning rate scheduler
            if update_successful:
                self.opt_param_scheduler.step(increment=args.global_batch_size)

            # Zero gradients after applying them
            for model_chunk in self.model:
                model_chunk.zero_grad_buffer()
            self.optimizer.zero_grad()

            # Update CPU weight cache after applying optimizer step
            self.update_cpu_params_dict(self.weights["actor"])

        return {
            "success": update_successful,
            "grad_norm": grad_norm if grad_norm is not None else 0.0,
        }

    def save_model(self, iteration: int) -> None:
        if self.args.debug_rollout_only:
            return

        save(iteration, self.model, self.optimizer, self.opt_param_scheduler)

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.offload and hasattr(mpu, "reload_process_groups"):
            mpu.reload_process_groups()

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())

        with torch_memory_saver.disable() if self.args.offload else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    print("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    for name in self.weights["old_actor"]:
                        self.weights["old_actor"][name].copy_(self.weights["rollout_actor"][name])
                    # Then copy current actor to rollout_actor
                    self.update_cpu_params_dict(self.weights["rollout_actor"])
                else:
                    self.update_cpu_params_dict(self.weights["old_actor"])

        if self.args.offload and hasattr(mpu, "destroy_process_groups"):
            mpu.destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights[model_tag] = {}
        self.update_cpu_params_dict(self.weights[model_tag])

    def connect_actor_critic(
        self,
        actor_handle: Optional[ActorHandle] = None,
        master_address: Optional[str] = None,
        master_port: Optional[int] = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )

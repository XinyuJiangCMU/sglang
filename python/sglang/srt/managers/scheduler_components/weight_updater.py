from __future__ import annotations

import logging
import struct  # KV_SCALE_PROBE only -- remove with the probe
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import msgspec
import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    BeginWeightUpdateReqOutput,
    ChecksumInfo,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    EndWeightUpdateReqInput,
    EndWeightUpdateReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    PullWeightsReqInput,
    PullWeightsReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils.weight_checker import overall_checksum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV_SCALE_PROBE -- temporary instrumentation, remove before merge.
#
# Observes layer.k_scale / layer.v_scale around the weights pause/resume cycle.
# Read-only: it never writes to the model and never raises into the caller.
# ---------------------------------------------------------------------------

# Incremented on both cycle entry points. A cycle can begin either with a weights
# pause (the colocate offload path) or straight at begin_weight_update (when
# --offload-rollout-level omits `weight`), so keying off the pause alone would
# attribute a whole cycle's phases to the previous cycle's number.
_kv_scale_cycle = 0


def _f32_bits(value: float) -> str:
    """Raw bit pattern. Structured residue (page tables, handles) is obvious in hex
    and invisible in decimal -- the shared-page signature was only found this way."""
    return f"0x{struct.unpack('<I', struct.pack('<f', value))[0]:08X}"


def _verdict(k: float, v: float) -> str:
    """Which of kv_cache.py's three branches this pair takes.

    NOTE the asymmetry in the third branch: `assert layer.k_scale > 0.0` only fires
    when k <= 0. A (k > 0, v <= 0) pair passes the assert and then duplicates
    max(k, v) = k into BOTH scales -- silent corruption, not a crash. Treating every
    mixed-sign pair as a crash overstates crashes and halves the silent-corruption
    count, which is the more dangerous outcome.
    """
    if k > 0.0 and v > 0.0:
        return "ok_1.0" if k == 1.0 else "silent_corrupt"
    if k <= 0.0 and v <= 0.0:
        return "default_1.0"
    if k > 0.0:
        return "silent_corrupt_dup"
    return "assert_crash"


def _log_kv_scale_probe(model, phase: str, role: str, cycle: int, rank: int) -> None:
    entries = [
        (name, param)
        for name, param in model.named_parameters()
        if name.endswith(("k_scale", "v_scale"))
    ]
    if not entries:
        logger.warning(
            "[KV_SCALE_PROBE] cycle=%d phase=%s role=%s rank=%d count=0",
            cycle, phase, role, rank,
        )
        return

    values = torch.stack([p.detach().reshape(()) for _, p in entries]).float().cpu()

    # Pair k with v per attention module: the three branches are decided by the pair,
    # not by either scale alone.
    by_module: Dict[str, Dict[str, float]] = {}
    for (name, _), value in zip(entries, values.tolist()):
        module_name, _, which = name.rpartition(".")
        by_module.setdefault(module_name, {})[which] = value

    verdicts: Dict[str, int] = {}
    dirty = []
    for module_name in sorted(by_module):
        pair = by_module[module_name]
        k, v = pair.get("k_scale"), pair.get("v_scale")
        if k is None or v is None:
            continue
        outcome = _verdict(k, v)
        verdicts[outcome] = verdicts.get(outcome, 0) + 1
        if outcome != "ok_1.0" and len(dirty) < 6:
            dirty.append(
                f"{module_name} k={k:.6g}[{_f32_bits(k)}] v={v:.6g}[{_f32_bits(v)}]"
            )

    # The host-side floats written by process_weights_after_loading (kv_cache.py:84-85).
    # They live in host memory, so pause/resume cannot touch them -- yet several
    # attention backends read *these* rather than the device tensor. When they and the
    # device tensor disagree, KV entries are stored scaled by one and read back by the
    # other. `vars()` because the attribute only exists once pwal has run.
    # `k_scale` is an nn.Parameter, so it lives in module._parameters, NOT in
    # vars(module); `k_scale_float` is a plain float and does land in vars(module).
    mirror = [
        (name, vars(module).get("k_scale_float"), vars(module).get("v_scale_float"))
        for name, module in model.named_modules()
        if "k_scale" in module._parameters
    ][:3]

    finite = torch.isfinite(values)
    finite_values = values[finite]
    logger.warning(
        "[KV_SCALE_PROBE] cycle=%d phase=%s role=%s rank=%d count=%d finite=%d "
        "eq1.0=%d zero=%d negative=%d min=%s max=%s verdict=%r mirror=%r dirty=[%s]",
        cycle, phase, role, rank, len(entries),
        int(finite.sum().item()),
        int((values == 1.0).sum().item()),
        int((values == 0).sum().item()),
        int((values < 0).sum().item()),
        f"{float(finite_values.min().item()):.6g}" if finite_values.numel() else "n/a",
        f"{float(finite_values.max().item()):.6g}" if finite_values.numel() else "n/a",
        verdicts, mirror, " | ".join(dirty),
    )


def _merge_checksum_payloads(role_payloads: List[Tuple[str, Dict]]) -> Dict:
    merged: Dict[str, str] = {}
    parallelism_infos = []
    for role, p in role_payloads:
        for name, chk in p["checksums"].items():
            # Only non-target roles are prefixed, so target keys stay stable.
            key = name if role == "" else f"{role}.{name}"
            if key in merged:
                raise ValueError(f"checksum key collision: {key}")
            merged[key] = chk
        parallelism_infos.append({"role": role or "target", **p["parallelism_info"]})
    return {
        "checksums": merged,
        "per_gpu_checksum": overall_checksum(merged),
        "parallelism_info": parallelism_infos,
    }


def _parse_runner_selector(selector: str) -> Set[str]:
    """Map a {target, draft, all} weight-op selector to the set of roles it covers."""
    if selector == "all":
        return {"target", "draft"}
    if selector in ("target", "draft"):
        return {selector}
    raise ValueError(
        f"invalid selector {selector!r}; expected 'target', 'draft', or 'all'"
    )


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    _weight_update_in_progress: bool = False
    _weight_update_loaded: bool = False
    # Runner selector for the open session, recorded at begin_weight_update and
    # reused by end_weight_update so the same set is restored and finalized.
    _weight_update_selector: str = "all"

    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )

    def flush_cache_after_weight_update(self, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        with self._observe_weight_load("disk"):
            success, message = self.tp_worker.update_weights_from_disk(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_disk(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success, message=message, num_paused_requests=0
            )

    def pull_weights(self, recv_req: PullWeightsReqInput):
        """Sync this host's local checkpoint up to recv_req.target_version.

        Every rank runs the pull; a per-host file lock collapses co-located
        ranks to one pull. Success is gathered across the TP group (all nodes),
        so the reply only reports success once every host holds a verified
        checkpoint.
        """
        from sglang.srt.weight_sync import local_checkpoint

        server_args = self.tp_worker.model_runner.server_args
        try:
            local_checkpoint.pull(
                local_checkpoint_dir=recv_req.local_checkpoint_dir,
                base_dir=server_args.model_path,
                source_dir=recv_req.source_dir,
                target_version=recv_req.target_version,
                pre_read_hook=server_args.custom_pull_weights_pre_read_hook,
            )
            success, message = True, "Success."
        except Exception:
            success, message = False, traceback.format_exc()
            logger.error(message)

        tp_size = (
            torch.distributed.get_world_size(group=self.tp_cpu_group)
            if torch.distributed.is_initialized()
            else 1
        )
        if tp_size > 1:
            results = [None] * tp_size
            torch.distributed.all_gather_object(
                results, (success, message), group=self.tp_cpu_group
            )
            success = all(ok for ok, _ in results)
            message = "; ".join(msg for ok, msg in results if not ok) or message
        return PullWeightsReqOutput(success=success, message=message)

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success=success, message=message)

    def destroy_weights_update_group(
        self,
        recv_req: DestroyWeightsUpdateGroupReqInput,
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success=success, message=message)

    def iter_weight_update_workers(
        self, selector: str = "all"
    ) -> List[Tuple[str, Any]]:
        """Resolve a {target, draft, all} selector to (role, worker) pairs, target
        first. This is the worker-level inclusion decision; each worker then
        contributes its own runners via iter_runners()."""
        parsed = _parse_runner_selector(selector)
        workers: List[Tuple[str, Any]] = []
        if "target" in parsed:
            workers.append(("target", self.tp_worker))
        if "draft" in parsed and self.draft_worker is not None:
            workers.append(("draft", self.draft_worker))
        return workers

    def get_model_runners(self, selector: str = "all") -> List[Tuple[str, Any]]:
        """Resolve a {target, draft, all} selector to (role, ModelRunner) pairs,
        target first. Role is "" for the target runner; draft roles come from the
        draft worker's iter_runners()."""
        runners: List[Tuple[str, Any]] = []
        for _, worker in self.iter_weight_update_workers(selector):
            runners += worker.iter_runners()
        return runners

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter, fanning out to the selected runners."""
        assert (
            self._weight_update_in_progress
        ), "update_weights_from_distributed requires an open begin_weight_update session"
        with self._observe_weight_load("distributed"):
            # Only the target (main) model joined this process's update group, so it
            # receives the broadcast once; the received weights are then loaded into
            # each selected runner locally. Draft runners never join the group.
            try:
                weights = self.tp_worker.model_runner.weight_updater.receive_weights_from_distributed(
                    recv_req.names,
                    recv_req.dtypes,
                    recv_req.shapes,
                    recv_req.group_name,
                    recv_req.load_format,
                )
                for _, runner in self.get_model_runners(recv_req.selector):
                    runner.weight_updater.load_weights(weights)
                success, message = True, "Succeeded to update parameter online."
            except Exception as e:
                success = False
                message = (
                    f"Failed to update parameter online: {e}. The full weights of the "
                    "ModelRunner are partially updated. Please discard the whole weights."
                )
                logger.error(message)
            if success:
                self._weight_update_loaded = True
                self.flush_cache_after_weight_update(recv_req)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors, fanning out to the
        selected runners."""
        assert (
            self._weight_update_in_progress
        ), "update_weights_from_tensor requires an open begin_weight_update session"
        with self._observe_weight_load("tensor"):
            monkey_patch_torch_reductions()
            named_tensors = MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_worker.ps.tp_rank]
            )
            success, message = True, "Success"
            for _, runner in self.get_model_runners(recv_req.selector):
                success, message = runner.weight_updater.update_weights_from_tensor(
                    named_tensors=named_tensors,
                    load_format=recv_req.load_format,
                )
                if not success:
                    break
            if success:
                self._weight_update_loaded = True
            if success:
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromTensorReqOutput(success=success, message=message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_ipc(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter=parameter)

    def _log_kv_scales(self, phase: str) -> None:
        """KV_SCALE_PROBE -- temporary, remove before merge."""
        global _kv_scale_cycle
        if phase in ("before_weights_pause", "before_begin_weight_update"):
            _kv_scale_cycle += 1
        try:
            rank = torch.distributed.get_rank(group=self.tp_cpu_group)
        except Exception:
            rank = -1
        # Deliberately NOT filtered to rank 0. Within a single engine, TP0 has been
        # observed holding -2.05e38 while TP1-3 read 0 at the same instant, so a
        # rank-0-only probe cannot tell "driver zeroed the page" from "the page holds
        # someone else's data" -- which is the whole question.
        try:
            for role, runner in self.get_model_runners("all"):
                _log_kv_scale_probe(
                    runner.model, phase, role or "target", _kv_scale_cycle, rank
                )
        except Exception as exc:
            logger.warning(
                "[KV_SCALE_PROBE] cycle=%d phase=%s rank=%d unavailable=%r",
                _kv_scale_cycle, phase, rank, exc,
            )

    def begin_weight_update(self, recv_req: BeginWeightUpdateReqInput):
        """Begin a weight-update session: restore in-place-packed weights to a
        loadable state on the selected runners (target and/or draft), so the draft
        model is prepared identically to the target. The selector is recorded and
        reused by end_weight_update so the same set is finalized."""
        assert (
            not self._weight_update_in_progress
        ), "begin_weight_update called while a weight-update session is already open"
        self._log_kv_scales("before_begin_weight_update")
        self._weight_update_selector = recv_req.selector
        for _, runner in self.get_model_runners(recv_req.selector):
            runner.begin_weight_update()
        self._weight_update_in_progress = True
        self._weight_update_loaded = False
        torch.distributed.barrier(group=self.tp_cpu_group)
        return BeginWeightUpdateReqOutput(success=True, message="Success")

    def end_weight_update(self, recv_req: EndWeightUpdateReqInput):
        """End the weight-update session on the runners begin_weight_update opened
        (its recorded selector): quant finalize on each, plus model.post_load_weights
        only when load_weights was bypassed this session (e.g. P2P/RDMA)."""
        assert (
            self._weight_update_in_progress
        ), "end_weight_update called without begin_weight_update"
        self._log_kv_scales("before_end_weight_update")
        run_post_load = not self._weight_update_loaded
        for _, runner in self.get_model_runners(self._weight_update_selector):
            runner.end_weight_update(run_post_load=run_post_load)
        # process_weights_after_loading runs inside end_weight_update: this is the pair
        # that shows what the three-branch logic actually did with the dirty values.
        self._log_kv_scales("after_end_weight_update")
        self._weight_update_in_progress = False
        torch.distributed.barrier(group=self.tp_cpu_group)
        return EndWeightUpdateReqOutput(success=True, message="Success")

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        scheduler = self.scheduler
        assert self.is_fully_idle(
            ignore_waiting=scheduler is not None and scheduler._engine_paused
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.release_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.release_memory_occupation()
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            # Ground truth for the cycle: everything downstream is compared against this.
            self._log_kv_scales("before_weights_pause")
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            # First look at the fresh physical pages, before anything writes to them.
            self._log_kv_scales("after_weights_resume_before_static_import")
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            # _export_static_state collects named_buffers() only, and these scales are
            # nn.Parameter -- so this pair should read identically. Logged to prove it
            # rather than argue it.
            self._log_kv_scales("after_static_import")
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.resume_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self, recv_req: CheckWeightsReqInput):
        try:
            role_payloads = []
            for role, runner in self.get_model_runners(recv_req.selector):
                p = runner.check_weights(
                    action=recv_req.action,
                    allow_quant_error=recv_req.allow_quant_error,
                    skip_tensor_list=recv_req.skip_tensor_list,
                )
                if p is not None:
                    role_payloads.append((role, p))
            payload = _merge_checksum_payloads(role_payloads) if role_payloads else None

            tp_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            if tp_size > 1 and payload is not None:
                all_payloads = [None] * tp_size
                torch.distributed.all_gather_object(
                    all_payloads, payload, group=self.tp_cpu_group
                )
                payload = all_payloads
            if payload is not None:
                # Normalize to one ChecksumInfo per rank so the wire shape is a
                # uniform List[ChecksumInfo] (tp==1 becomes a single-element list).
                per_rank = payload if isinstance(payload, list) else [payload]
                payload = [msgspec.convert(p, ChecksumInfo) for p in per_rank]
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self, params):
        url = params["url"]

        self.tp_worker.model_runner.weight_exporter.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.weight_exporter.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        self.tp_worker.model_runner.weight_exporter.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ],
        # skipped_params=[
        #     (name, param.detach().to("cpu", copy=True))
        #     for name, param in model.named_parameters()
        #     if getattr(param, "_skip_weight_check", False)
        # ],
    )


def _import_static_state(model, static_params):
    with torch.inference_mode():
        # for key, live in (
        #     ("buffers", dict(model.named_buffers())),
        #     ("skipped_params", dict(model.named_parameters())),
        # ):
        #     for name, tensor in static_params.get(key, []):
        #         target = live.get(name)
        #         if target is not None:
        #             target.copy_(tensor)
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor

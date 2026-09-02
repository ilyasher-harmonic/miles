from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import ray
from ray.actor import ActorHandle

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.workers.backend_capability import factory
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import ClusterBackend, WorkerCommBackend

logger = logging.getLogger(__name__)


def launch_worker_manager(args):
    match ClusterBackend(args.cluster_backend):
        case ClusterBackend.KUBERNETES:
            return None
        case ClusterBackend.RAY:
            return _launch_ray_worker_manager(args)


@asynccontextmanager
async def shutting_down_worker_manager(worker_manager_handle: ActorHandle | None) -> AsyncIterator[None]:
    """Own the manager for the length of a driver, releasing it on the failing exit paths too.

    A shutdown that fails while another failure is already leaving is logged rather than raised, so
    the run still reports the failure that caused the teardown instead of the one it ran into.
    """
    try:
        yield
    except BaseException:
        try:
            await shutdown_worker_manager(worker_manager_handle)
        except Exception:
            logger.exception("shutting the worker manager down failed; reporting the failure that was already leaving")
        raise
    await shutdown_worker_manager(worker_manager_handle)


async def shutdown_worker_manager(worker_manager_handle: ActorHandle | None) -> None:
    if worker_manager_handle is None:
        return
    try:
        await worker_manager_handle.shutdown.remote()
    finally:
        ray.kill(worker_manager_handle)


def get_backend_capability(args) -> BackendCapability:
    return factory.get_backend_capability(
        specs=compute_specs(args), cluster_backend=ClusterBackend(args.cluster_backend)
    )


def _launch_ray_worker_manager(args):
    from miles.ray.placement_group import create_placement_groups

    specs = compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(args, specs, pgs, comm_backend=WorkerCommBackend(args.worker_comm_backend))

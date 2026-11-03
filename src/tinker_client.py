"""Tinker sampling clients from a model spec: a base model id or a tinker:// checkpoint path."""

from __future__ import annotations

import asyncio

import tinker


def is_checkpoint(spec: str) -> bool:
    return spec.startswith("tinker://")


def sampling_client(spec: str):
    service = tinker.ServiceClient()
    if is_checkpoint(spec):
        return service.create_sampling_client(model_path=spec)
    return service.create_sampling_client(base_model=spec)


def base_model_of(spec: str) -> str:
    """The base model a checkpoint was trained from; a base model id is returned unchanged."""
    if not is_checkpoint(spec):
        return spec
    rest = tinker.ServiceClient().create_rest_client()
    run = asyncio.run(rest.get_training_run_by_tinker_path_async(spec))
    return run.base_model

"""Explicit compatibility boundary for an already configured MLflow client.

MLflow 3.14 local SQL search exposes base64 pagination tokens as bytes, although
APF's transport contract uses strings. Missing runs are exposed via MLflow's
structured RESOURCE_DOES_NOT_EXIST code, not Python's FileNotFoundError.
No URI discovery, global configuration, experiment creation, retries, logging,
run termination, scientific identity changes or authentication inference occur.
"""
from dataclasses import dataclass, field
from typing import Any


class _ResearchRunPage(list):
    def __init__(self, values, token):
        super().__init__(values)
        self.token = token


@dataclass(frozen=True)
class ResearchMlflowClient:
    """Delegate only search/create/read to an explicitly configured client.

    Wrap with BoundMlflowClient(destination, ResearchMlflowClient(real_client)).
    Returned Run objects, tags, IDs and run statuses are never modified.
    Only ASCII token representation and structured missing-run exceptions are
    translated. Other errors retain their existing service-level handling.
    """
    client: Any = field(repr=False)

    def search_runs(self, *args, **kwargs):
        page = self.client.search_runs(*args, **kwargs)
        if not isinstance(page, (list, tuple)):
            raise ValueError("unsupported_mlflow_search_page")
        token = getattr(page, "token", None)
        if type(token) is bytes:
            try:
                token = token.decode("ascii")
            except UnicodeDecodeError:
                raise ValueError("unsupported_mlflow_page_token") from None
        if token is not None and (type(token) is not str or not 0 < len(token) <= 4096):
            raise ValueError("unsupported_mlflow_page_token")
        return _ResearchRunPage(page, token)

    def create_run(self, *args, **kwargs):
        return self.client.create_run(*args, **kwargs)

    def get_run(self, *args, **kwargs):
        try:
            return self.client.get_run(*args, **kwargs)
        except Exception as error:
            # Match the documented structured code, never exception text.
            if getattr(error, "error_code", None) == "RESOURCE_DOES_NOT_EXIST":
                raise FileNotFoundError("research_remote_run_not_found") from None
            raise

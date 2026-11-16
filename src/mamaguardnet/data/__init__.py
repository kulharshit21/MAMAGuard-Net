"""Dataset inventory, manifests, split policy, and external-test protection."""

from mamaguardnet.data.cases import CaseRecord, read_case_manifest, write_case_manifest

__all__ = ["CaseRecord", "read_case_manifest", "write_case_manifest"]

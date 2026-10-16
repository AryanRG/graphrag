# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for loading incremental-update documents."""

from collections.abc import AsyncIterator

import pandas as pd
import pytest
from graphrag.index.workflows.load_update_documents import load_update_documents
from graphrag_input import InputReader, TextDocument
from graphrag_storage import StorageConfig, StorageType, create_storage
from graphrag_storage.tables.parquet_table_provider import ParquetTableProvider


class _StaticInputReader(InputReader):
    """Yield an in-memory set of input documents."""

    def __init__(self, documents: list[TextDocument]) -> None:
        self._documents = documents

    async def _iterate_files(self) -> AsyncIterator[TextDocument]:
        for document in self._documents:
            yield document

    async def read_file(self, path: str) -> list[TextDocument]:
        """Return no path-based documents; iteration uses the supplied documents."""
        return []


def _document(title: str) -> TextDocument:
    return TextDocument(
        id=f"id-{title}",
        title=title,
        text=f"text for {title}",
        creation_date="2026-01-01",
    )


async def _previous_provider(*titles: str) -> ParquetTableProvider:
    storage = create_storage(StorageConfig(type=StorageType.Memory))
    provider = ParquetTableProvider(storage)
    await provider.write_dataframe(
        "documents",
        pd.DataFrame([
            {
                **vars(_document(title)),
                "human_readable_id": index,
            }
            for index, title in enumerate(titles)
        ]),
    )
    return provider


async def test_rejects_deleted_input_documents() -> None:
    """A removal must not be reported as a successful no-op update."""
    previous_provider = await _previous_provider("a.txt", "b.txt")
    input_reader = _StaticInputReader([_document("a.txt")])

    with pytest.raises(
        ValueError,
        match="Incremental updates do not support deleted input documents",
    ):
        await load_update_documents(input_reader, previous_provider)


async def test_returns_only_new_input_documents() -> None:
    """Additions remain supported when no prior document was removed."""
    previous_provider = await _previous_provider("a.txt")
    input_reader = _StaticInputReader([_document("a.txt"), _document("b.txt")])

    result = await load_update_documents(input_reader, previous_provider)

    assert result["title"].tolist() == ["b.txt"]

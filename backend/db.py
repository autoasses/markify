from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

logger = logging.getLogger("autoassess")

# ============================================================
# Configuration
# ============================================================
#
# Everything is driven by env vars so nothing is hard-coded:
#
#   MONGODB_URI                 connection string (default: local mongod)
#   MONGODB_DB                  database name      (default: "autoassess")
#   MONGODB_SCANNED_COLLECTION  collection for OCR'd (scanned) documents
#                                (default: "scanned_documents")
#   MONGODB_MODEL_ANSWERS_COLLECTION
#                                collection for documents that went through
#                                plain markdown conversion, no OCR needed
#                                (default: "model_answers")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "autoassess")
SCANNED_DOCUMENTS_COLLECTION = os.getenv(
    "MONGODB_SCANNED_COLLECTION", "scanned_documents"
)
MODEL_ANSWERS_COLLECTION = os.getenv(
    "MONGODB_MODEL_ANSWERS_COLLECTION", "model_answers"
)

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


# ============================================================
# Lifecycle
# ============================================================

def connect() -> None:
    """Open the MongoDB connection. Call once on app startup."""

    global _client, _db

    logger.info("Connecting to MongoDB at %s", MONGODB_URI)

    _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)

    # Fail fast (and loudly) if MongoDB is unreachable, instead of only
    # discovering it later when the first insert silently times out.
    try:
        _client.admin.command("ping")
    except PyMongoError:
        logger.exception("Could not reach MongoDB at %s", MONGODB_URI)
        raise

    _db = _client[MONGODB_DB]

    # Helpful, non-unique indexes for the common lookups on each collection.
    for collection_name in (SCANNED_DOCUMENTS_COLLECTION, MODEL_ANSWERS_COLLECTION):
        collection = _db[collection_name]
        collection.create_index("filename")
        collection.create_index("created_at")
        collection.create_index("type")

    logger.info(
        "MongoDB connected: db=%s scanned_collection=%s model_answers_collection=%s",
        MONGODB_DB,
        SCANNED_DOCUMENTS_COLLECTION,
        MODEL_ANSWERS_COLLECTION,
    )


def close() -> None:
    """Close the MongoDB connection. Call once on app shutdown."""

    global _client, _db

    if _client is not None:
        _client.close()

    _client = None
    _db = None


def get_collection(name: str) -> Collection:
    if _db is None:
        raise RuntimeError(
            "MongoDB is not connected. Call db.connect() on startup first."
        )

    return _db[name]


# ============================================================
# Writes
# ============================================================

def save_document(payload: dict[str, Any], collection_name: str) -> Optional[str]:
    """
    Insert one processed-document record into the given collection.

    Returns the inserted document's string id, or None if the write
    failed. Failures are logged but never raised, so a MongoDB outage
    does not take down the OCR/segmentation endpoint - the caller
    decides whether a missing id should be surfaced to the client.
    """

    record = dict(payload)
    record.setdefault("created_at", datetime.now(timezone.utc))

    try:
        collection = get_collection(collection_name)
        result = collection.insert_one(record)
        return str(result.inserted_id)

    except PyMongoError:
        logger.exception(
            "Failed to save document to MongoDB collection '%s'.",
            collection_name,
        )
        return None
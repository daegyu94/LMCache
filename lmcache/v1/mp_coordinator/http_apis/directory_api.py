# SPDX-License-Identifier: Apache-2.0
"""Key-directory endpoints on the coordinator (fleet-level).

The ``/directory`` surface, thin over the :class:`KeyDirectory` carried
on the typed :class:`CoordinatorContext`: placement lookup, fragment
(blend) lookup, key/token-id listing, and stats — all read-only. The
directory is written by the cache-event stream, which arrives on
``POST /events`` (see :mod:`events_api`). See
``docs/design/v1/mp_coordinator/views/key_directory.md``.
"""

# Standard
import asyncio

# Third Party
from fastapi import APIRouter, HTTPException, Query, Request

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.api import BlendMatch, BlendNamespace
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.l1_digest_index import L1DigestView
from lmcache.v1.mp_coordinator.schemas import (
    BlendLookupRequest,
    BlendLookupResponse,
    BlendMatchModel,
    DirectoryKeyInfo,
    DirectoryKeyPlacements,
    DirectoryListResponse,
    DirectoryLookupRequest,
    DirectoryLookupResponse,
    decode_tokens,
)
from lmcache.v1.mp_coordinator.views.key_directory import DirectoryStats, KeyDirectory
from lmcache.v1.multiprocess.cache_control.key_resolver import resolve_object_keys

router = APIRouter()


def _parse_known_digests(known: str) -> dict[str, tuple[int, int]]:
    """Parse ``instance_id:generation:count,...`` into a map.

    Raises:
        ValueError: If an entry is not three colon-separated fields with
            integer generation and count.
    """
    held: dict[str, tuple[int, int]] = {}
    for item in filter(None, known.split(",")):
        instance_id, generation, count = item.rsplit(":", 2)
        held[instance_id] = (int(generation), int(count))
    return held


def _encode_digest(
    view: L1DigestView, held: tuple[int, int] | None
) -> dict[str, object]:
    """Encode ``view`` as a delta against ``held`` when that is smaller."""
    entry: dict[str, object] = {"generation": view.generation, "count": view.count}
    if held is not None and held[0] == view.generation and held[1] <= view.count:
        added = view.added[held[1] : view.count]
        # Compare wire sizes: a hash is hex (2x) plus JSON quoting and a comma;
        # base64 adds a third to the bits.
        if sum(2 * len(h) + 4 for h in added) < len(view.digest.bits) * 4 // 3:
            entry["added"] = [h.hex() for h in added]
            return entry
    entry.update(view.digest.to_json())
    return entry


@router.post("/directory/lookup")
async def lookup_placements(
    body: DirectoryLookupRequest, request: Request
) -> DirectoryLookupResponse:
    """Resolve keys — or a request's token sequence — to placements.

    The tokens form hashes ``token_ids`` with the fleet's token hasher
    and expands each complete chunk into its per-rank object keys.
    Chunk hashes are prefix-chained, so ``token_ids`` must be the
    request's full sequence from position 0; trailing incomplete chunks
    are ignored.

    Args:
        body: Either the keys to resolve or the token sequence and its
            key-resolution parameters.

    Returns:
        Chunk count plus one result per resolved key, in request order,
        each with its known placements, the chunk's token ids, and the
        key's access count. Unknown keys get empty lists and ``0``.

    Raises:
        HTTPException: 400 when the token sequence exceeds the
            per-request cap or a key field is invalid.
    """
    ctx = get_context(request)
    directory = ctx.views.get(KeyDirectory)
    if body.keys:
        encoded_keys = list(body.keys)
        obj_keys = [encoded.to_object_key() for encoded in encoded_keys]
        chunks = len(obj_keys)
    else:
        try:
            obj_keys, chunks = resolve_object_keys(
                token_hasher=ctx.token_hasher,
                model_name=body.model_name,
                world_size=body.world_size,
                token_ids=body.token_ids,
                cache_salt=body.cache_salt,
                tags=body.tags,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        obj_keys = directory.get_keys_across_object_groups(obj_keys)
        encoded_keys = [key.to_encoded_object_key() for key in obj_keys]
    placements = directory.lookup(obj_keys)
    keys_with_placements = [
        key
        for key, key_placements in zip(obj_keys, placements, strict=True)
        if key_placements
    ]
    # By tokens, a chunk counts once across its ranks and object groups.
    chunks_found = (
        len(keys_with_placements)
        if body.keys
        else len({key.chunk_hash for key in keys_with_placements})
    )
    directory.record_lookup(
        "prefix", chunks_requested=chunks, chunks_found=chunks_found
    )
    token_ids = directory.get_token_ids([key.chunk_hash for key in obj_keys])
    access_counts = directory.get_access_counts(obj_keys)
    return DirectoryLookupResponse(
        chunks=chunks,
        results=[
            DirectoryKeyPlacements(
                key=encoded,
                placements=key_placements,
                token_ids=list(tokens),
                access_count=access_count,
            )
            for encoded, key_placements, tokens, access_count in zip(
                encoded_keys, placements, token_ids, access_counts, strict=True
            )
        ],
    )


@router.post("/directory/blend-lookup")
async def blend_lookup(
    body: BlendLookupRequest, request: Request
) -> BlendLookupResponse:
    """Find cached chunk content anywhere inside a query sequence.

    The fragment counterpart to ``/directory/lookup``: the query need not
    be a prefix, and each match reports both where the content sits in
    the query and where it sat when stored, so the caller can re-RoPE it.

    Matches are restricted to the namespace the body names.

    Args:
        body: The query tokens and the caller's key-resolution
            parameters.
        request: The FastAPI request carrying the coordinator context.

    Returns:
        Matched chunks, ascending by query position.
    """
    ctx = get_context(request)
    directory = ctx.views.get(KeyDirectory)
    tokens = decode_tokens(body.tokens_b64)
    namespace = BlendNamespace(
        model_name=body.model_name,
        cache_salt=body.cache_salt,
        world_size=body.world_size,
        tags=body.tags,
    )

    chunk_size = ctx.token_hasher.chunk_size

    def _match() -> BlendLookupResponse:
        """Run the fragment match, count it, and shape it for the wire."""
        matches = directory.blend_match(tokens, namespace)
        directory.record_lookup(
            "non_prefix",
            chunks_requested=len(tokens) // chunk_size,
            chunks_found=_covered_chunks(matches, chunk_size),
        )
        return BlendLookupResponse(
            matches=[
                BlendMatchModel(
                    chunk_hash=match.chunk_hash.hex(),
                    old_st=match.old_st,
                    cur_st=match.cur_st,
                )
                for match in matches
            ]
        )

    # The rolling hash walks the whole query — keep it off the event loop.
    return await asyncio.to_thread(_match)


@router.get("/directory/keys")
async def list_directory_keys(
    request: Request,
    tier: Tier = Tier.ALL,
    instance_id: str = "",
    backend: str = "",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=10000),
) -> DirectoryListResponse:
    """List directory keys and their placements, one page at a time.

    A snapshot for inspection: pages of a changing directory may skip
    or repeat keys.

    Args:
        request: The FastAPI request carrying the coordinator context.
        tier: Keep placements on this tier (``all`` keeps every tier).
        instance_id: Keep placements reported by this instance (empty
            keeps every instance).
        backend: Keep placements on this backend (empty keeps every
            backend).
        offset: Matching keys to skip.
        limit: Maximum keys to return.

    Returns:
        The number of keys matching the filters plus the requested page,
        each key with its matching placements, the number of token ids
        known for its chunk, and its access count.
    """
    directory = get_context(request).views.get(KeyDirectory)

    def _scan() -> DirectoryListResponse:
        """Page the directory and shape the rows for the wire."""
        total, page = directory.list_keys(tier, instance_id, backend, offset, limit)
        token_ids = directory.get_token_ids([key.chunk_hash for key in page])
        access_counts = directory.get_access_counts(list(page))
        return DirectoryListResponse(
            total=total,
            keys=[
                DirectoryKeyInfo(
                    key=key.to_encoded_object_key(),
                    placements=placements,
                    num_tokens=len(tokens),
                    access_count=access_count,
                )
                for (key, placements), tokens, access_count in zip(
                    page.items(), token_ids, access_counts, strict=True
                )
            ],
        )

    # ``total`` walks every matching record — keep the scan off the event loop.
    return await asyncio.to_thread(_scan)


@router.get("/directory/l1-digests")
async def l1_digests(
    request: Request, known: str = ""
) -> dict[str, dict[str, dict[str, object]]]:
    """Return a Bloom digest of each instance's L1 chunk hashes.

    P2P requesters poll this to skip lookup RPCs to peers that hold none of
    a request's chunks. A requester lists the digests it already holds, and
    for those only the hashes added since are sent, which keeps a poll small
    when filters are large and change little.

    Args:
        request: The FastAPI request carrying the coordinator context.
        known: Comma-separated ``instance_id:generation:count`` entries for
            the digests the caller already holds.

    Returns:
        ``{"digests": {instance_id: entry}}``. Every entry has
        ``generation`` and ``count``; a full one adds ``num_bits`` and
        ``bits`` (see :meth:`BloomDigest.to_json`), a delta adds ``added``,
        the hex chunk hashes to insert into the caller's copy.

    Raises:
        HTTPException: 400 if ``known`` is malformed.
    """
    try:
        held = _parse_known_digests(known)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    directory = get_context(request).views.get(KeyDirectory)

    def _encode() -> dict[str, dict[str, object]]:
        return {
            instance_id: _encode_digest(view, held.get(instance_id))
            for instance_id, view in directory.l1_digests().items()
        }

    digests = await asyncio.to_thread(_encode)
    return {"digests": digests}


@router.get("/directory/stats")
async def directory_stats(request: Request) -> DirectoryStats:
    """Return a point-in-time summary of directory contents.

    Returns:
        Key/placement counts, per-instance L1 key counts, and the
        blend-index counts. Per-emitter stream state (incarnation,
        applied seq, gap flag) lives on the ingest gate and is not
        exposed yet.
    """
    return get_context(request).views.get(KeyDirectory).stats()


def _covered_chunks(matches: list[BlendMatch], chunk_size: int) -> int:
    """Return how many chunks' worth of the query the matches cover.

    Matches can overlap in the query (two stored chunks sharing tokens), so
    their count can exceed the chunks the query holds; the tokens they
    cover together, overlaps counted once, cannot.

    Args:
        matches: Matches in ascending ``cur_st`` order, each covering
            ``chunk_size`` query tokens from ``cur_st``.
        chunk_size: Tokens per chunk.

    Returns:
        Covered query tokens divided by ``chunk_size``, rounded down.
    """
    covered_tokens = 0
    covered_until = 0
    for match in matches:
        start = max(match.cur_st, covered_until)
        end = match.cur_st + chunk_size
        if end > start:
            covered_tokens += end - start
            covered_until = end
    return covered_tokens // chunk_size

# SPDX-License-Identifier: Apache-2.0
"""Protocol definitions for per-client L2 QoS metadata."""

# First Party
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

REQUEST_NAMES = ["REGISTER_QOS_PROFILE"]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """Return the QoS handshake protocol definition."""
    return {
        "REGISTER_QOS_PROFILE": ProtocolDefinition(
            payload_classes=[str, int],
            response_class=None,
            handler_type=HandlerType.SYNC,
        )
    }

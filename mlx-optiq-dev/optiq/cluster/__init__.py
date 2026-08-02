"""OptiQ cluster: zero-config multi-Mac discovery + distributed inference.

`pip install mlx-optiq` on two Macs, `optiq cluster up` on each, and
`optiq cluster peers` from either sees the others over Thunderbolt/network —
no IPs, no hostfiles, no SSH-key dance.
"""

from .discovery import Peer, discover, advertise, node_txt, DEFAULT_PORT, SERVICE

__all__ = ["Peer", "discover", "advertise", "node_txt", "DEFAULT_PORT", "SERVICE"]

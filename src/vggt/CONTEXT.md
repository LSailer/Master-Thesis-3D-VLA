# VGGT context

Feature Extractor language for the streaming VGGT integration. Read the shared
glossary at the repo root first.

## Language

**Camera Token**:
The aggregator's learned per-frame summary token, occupying the first token
slot of each frame. It attends over everything the stream has retained, and
the camera head derives its pose estimate from it.
_Avoid_: camera position, pose token, camera point

**Register Tokens**:
The four attention scratch tokens following the Camera Token. They exist to
absorb high-norm attention artifacts during aggregation and are never consumed
downstream.
_Avoid_: extra tokens, spare tokens

**Patch Tokens**:
The per-frame tokens carrying spatial image content, one per 14×14 image
patch on the 37×37 grid.
_Avoid_: image tokens, grid features

**Global Half**:
The half of each full-width aggregator token produced by the global-attention
blocks (the alternating cross-frame path), as opposed to the frame half from
the within-frame path.
_Avoid_: global embedding, second half

**Global Patch Tokens**:
The Global Half of the Patch Tokens only — Camera and Register Tokens
excluded. The scene-content representation of one streamed frame.
_Avoid_: global tokens (ambiguous: may include special tokens), patch features

**Position Signal**:
The per-step observation component that tells the agent where it currently is
within the house. Realized by the Camera Token in the global-embedding house
context; realized by the camera pose in the points-based house context.
_Avoid_: camera pose (when the Camera Token is meant), location feature

**Point-Cloud Snapshot**:
A diagnostic export of the current generated point cloud with colour
information, written for later visual inspection. Not consumed by training.
_Avoid_: house map dump, debug points

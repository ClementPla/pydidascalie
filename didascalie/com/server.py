import traceback
import numpy as np
import zmq
import msgpack

from . import decorators, masks

# 2: segmentation ops, and `functions` in the ping reply.
PROTOCOL_VERSION = 2

# The sequence being assembled for a `register_sequence_seg` call. Frames
# arrive one message at a time so a long video never has to fit in one.
_SESSION: dict = {}

def _decode_image(buf: bytes, shape: tuple, dtype: str) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape)

def _decode_masks(payload: dict | None) -> dict[str, np.ndarray]:
    return {
        name: _decode_image(m["buf"], m["shape"], m["dtype"])
        for name, m in (payload or {}).items()
    }


def _missing(name: str, kind: str) -> dict:
    return {"ok": False, "error": f"No function registered as {name!r}. "
                                  f"Available: {decorators._list(kind)}"}


def _extras(entry, msg: dict, current_masks) -> dict:
    """The keyword arguments `entry` declared, from the request."""
    available = {
        "masks": current_masks,
        "labels": list(msg["labels"]),
        "active_label": msg.get("active_label"),
        "frame_index": msg.get("frame_index"),
    }
    return {k: (v() if callable(v) else v) for k, v in available.items() if k in entry.wants}


def _segment(msg: dict) -> dict:
    name = msg["name"]
    entry = decorators._get(name, decorators.SEG)
    if entry is None:
        return _missing(name, decorators.SEG)

    labels = list(msg["labels"])
    # Copied: frombuffer arrays are read-only, which torch.from_numpy rejects.
    image = _decode_image(msg["image"]["buf"], msg["image"]["shape"], msg["image"]["dtype"]).copy()
    hw = image.shape[:2]

    def current_masks():
        sent = _decode_masks(msg.get("masks"))
        return {l: sent[l].copy() if l in sent else np.zeros(hw, np.uint8) for l in labels}

    result = entry.fn(image, **_extras(entry, msg, current_masks))
    layers, unknown = masks.frame_layers(result, labels)
    return {"ok": True, "masks": masks.encode(layers, 0, hw), "unknown": unknown}


def _seq_begin(msg: dict) -> dict:
    name = msg["name"]
    if decorators._get(name, decorators.SEQUENCE_SEG) is None:
        return _missing(name, decorators.SEQUENCE_SEG)
    _SESSION.clear()
    _SESSION.update(msg=msg, frames=[None] * msg["n_frames"],
                    masks=[None] * msg["n_frames"], layers=None, shapes=None)
    return {"ok": True}


def _seq_frame(msg: dict) -> dict:
    i = msg["index"]
    img = msg["image"]
    _SESSION["frames"][i] = _decode_image(img["buf"], img["shape"], img["dtype"])
    _SESSION["masks"][i] = _decode_masks(msg.get("masks"))
    return {"ok": True}


def _seq_run() -> dict:
    begin = _SESSION["msg"]
    entry = decorators._get(begin["name"], decorators.SEQUENCE_SEG)
    labels = list(begin["labels"])
    frames = _SESSION["frames"]
    if any(f is None for f in frames):
        return {"ok": False, "error": "Sequence incomplete: some frames were never received."}
    shapes = [f.shape[:2] for f in frames]
    same_size = len(set(shapes)) <= 1

    def current_masks():
        per_label = {}
        for l in labels:
            stack = [m[l] if l in m else np.zeros(hw, np.uint8)
                     for m, hw in zip(_SESSION["masks"], shapes)]
            per_label[l] = np.stack(stack) if same_size else stack
        return per_label

    # Stacking copies, so the result is writable; a ragged sequence stays a list.
    stacked = np.stack(frames) if same_size and frames else [f.copy() for f in frames]
    extras = _extras(entry, begin, current_masks)
    _SESSION["frames"] = _SESSION["masks"] = None  # the function owns them now

    result = entry.fn(stacked, **extras)
    layers, unknown = masks.sequence_layers(result, labels, len(shapes))
    _SESSION.update(layers=layers, shapes=shapes)
    return {"ok": True, "unknown": unknown}


def _seq_result(msg: dict) -> dict:
    if _SESSION.get("layers") is None:
        return {"ok": False, "error": "No sequence result to fetch."}
    i = msg["index"]
    return {"ok": True, "masks": masks.encode(_SESSION["layers"], i, _SESSION["shapes"][i])}


def _handle(msg: dict) -> dict:
    op = msg.get("op")

    if op == "ping":
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            # Keypoint functions only: what a protocol-1 Didascalie lists.
            "registered": decorators._list(decorators.KEYPOINTS),
            "functions": decorators._describe(),
        }

    seg_ops = {
        "segment": lambda: _segment(msg),
        "seq_begin": lambda: _seq_begin(msg),
        "seq_frame": lambda: _seq_frame(msg),
        "seq_run": _seq_run,
        "seq_result": lambda: _seq_result(msg),
    }
    if op in seg_ops:
        try:
            return seg_ops[op]()
        except Exception as e:
            print(f"Error in {op!r}:", traceback.format_exc())
            # The last line is what fits in a toast; the traceback is above.
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    if op == "seq_end":
        _SESSION.clear()
        return {"ok": True}

    if op == "find_keypoints":
        name = msg["name"]
        entry = decorators._get(name, decorators.KEYPOINTS)
        if entry is None:
            return _missing(name, decorators.KEYPOINTS)
        fn = entry.fn
        try:
            ref = _decode_image(msg["ref"]["buf"], msg["ref"]["shape"], msg["ref"]["dtype"])
            mov = _decode_image(msg["mov"]["buf"], msg["mov"]["shape"], msg["mov"]["dtype"])
            existing = [
                ((float(p[0][0]), float(p[0][1])), (float(p[1][0]), float(p[1][1])))
                for p in msg.get("existing", [])
            ]
            result = fn(ref, mov, existing)
            pairs = [
                [[float(r[0]), float(r[1])], [float(m[0]), float(m[1])]]
                for (r, m) in result
            ]
            return {"ok": True, "pairs": pairs}
        except Exception:
            print(f"Error in {name!r}:", traceback.format_exc())
            return {"ok": False, "error": traceback.format_exc()}

    return {"ok": False, "error": f"Unknown op: {op!r}"}

def serve(port: int = 5556, host: str = "tcp://*"):
    """
    Block the current cell/thread and serve until Ctrl+C.

    5556 is where Didascalie looks by default (it uses 5555 itself).
    """
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"{host}:{port}")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    print(f"[didascalie] listening on {host}:{port}, "
          f"registered: {decorators._list()}")
    try:
        while True:
            try:
                events = dict(poller.poll(timeout=200))  # ms
            except KeyboardInterrupt:
                raise
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break
                raise

            if sock not in events:
                continue  # tick — lets Ctrl+C through

            raw = sock.recv()
            try:
                msg = msgpack.unpackb(raw, raw=False)
            except Exception as e:
                sock.send(msgpack.packb({"ok": False, "error": f"bad msgpack: {e}"}))
                continue

            reply = _handle(msg)
            sock.send(msgpack.packb(reply, use_bin_type=True))

    except KeyboardInterrupt:
        print("[didascalie] stopped")
    finally:
        sock.close(linger=0)
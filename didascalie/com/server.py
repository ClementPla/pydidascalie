import io
import traceback
import numpy as np
import zmq
import msgpack

from . import decorators

PROTOCOL_VERSION = 1

def _decode_image(buf: bytes, shape: tuple, dtype: str) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape)

def _handle(msg: dict) -> dict:
    op = msg.get("op")

    if op == "ping":
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "registered": decorators._list(),
        }

    if op == "find_keypoints":
        name = msg["name"]
        fn = decorators._get(name)
        if fn is None:
            return {"ok": False, "error": f"No function registered as {name!r}. "
                                          f"Available: {decorators._list()}"}
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

def serve(port: int = 5555, host: str = "tcp://*"):
    """
    Block the current cell/thread and serve until Ctrl+C.
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
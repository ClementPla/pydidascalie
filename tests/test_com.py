"""Protocol-level tests of the Didascalie bridge, driven through `_handle`."""
import msgpack
import numpy as np
import pytest

from didascalie.com import (
    Boolean,
    Enum,
    Float,
    Int,
    Result,
    decorators,
    register_kpts,
    register_seg,
    register_sequence_seg,
)
from didascalie.com.server import _handle

LABELS = ["vessel", "disc"]
H, W = 4, 6


def call(msg: dict) -> dict:
    """Round-trip through msgpack, as the socket would."""
    reply = _handle(msgpack.unpackb(msgpack.packb(msg, use_bin_type=True), raw=False))
    return msgpack.unpackb(msgpack.packb(reply, use_bin_type=True), raw=False)


def payload(arr: np.ndarray) -> dict:
    return {"buf": arr.tobytes(), "shape": list(arr.shape), "dtype": str(arr.dtype)}


def mask_of(m: dict) -> np.ndarray:
    return np.frombuffer(m["buf"], np.uint8).reshape(m["shape"])


def segment(name: str, **extra) -> dict:
    image = np.zeros((H, W, 3), np.uint8)
    return call({"op": "segment", "name": name, "image": payload(image),
                 "labels": LABELS, "active_label": "disc", "frame_index": 2, **extra})


def run_sequence(name: str, n: int = 3, masks=None) -> tuple[dict, list[dict]]:
    assert call({"op": "seq_begin", "name": name, "n_frames": n, "labels": LABELS,
                 "active_label": "disc", "frame_index": 1})["ok"]
    for i in range(n):
        frame = {"op": "seq_frame", "index": i,
                 "image": payload(np.full((H, W, 3), i, np.uint8))}
        if masks is not None:
            frame["masks"] = masks[i]
        assert call(frame)["ok"]
    run = call({"op": "seq_run"})
    results = [call({"op": "seq_result", "index": i}) for i in range(n)] if run["ok"] else []
    return run, results


@pytest.fixture(autouse=True)
def clean_registry():
    decorators._REGISTRY.clear()
    yield
    decorators._REGISTRY.clear()


def test_ping_describes_every_kind():
    @register_kpts("match")
    def match(ref, mov, existing):
        return []

    @register_seg
    def unet(image, masks, active_label):
        """Segments vessels.

        Longer explanation."""

    @register_sequence_seg("track")
    def track(frames):
        pass

    reply = call({"op": "ping"})
    assert reply["protocol_version"] == 2
    assert reply["registered"] == ["match"]
    by_name = {f["name"]: f for f in reply["functions"]}
    assert by_name["match"]["kind"] == "keypoints"
    assert by_name["unet"] == {"name": "unet", "kind": "seg", "doc": "Segments vessels.",
                               "wants": ["masks", "active_label"], "params": []}
    assert by_name["track"]["kind"] == "sequence_seg"


def test_register_alias_still_registers_keypoints():
    from didascalie.com import register

    with pytest.warns(DeprecationWarning):
        @register("old")
        def old(ref, mov, existing):
            return [((1, 2), (3, 4))]

    img = payload(np.zeros((2, 2, 3), np.uint8))
    reply = call({"op": "find_keypoints", "name": "old", "ref": img, "mov": img, "existing": []})
    assert reply == {"ok": True, "pairs": [[[1.0, 2.0], [3.0, 4.0]]]}


def test_unfillable_parameter_is_rejected_at_decoration():
    with pytest.raises(TypeError, match="threshold"):
        @register_seg
        def bad(image, threshold):
            pass

    @register_seg
    def fine(image, threshold=0.5):
        pass


def test_dict_result_targets_named_labels():
    @register_seg
    def f(image):
        return {"vessel": np.ones((H, W), bool), "nope": np.ones((H, W), bool), 1: np.zeros((H, W))}

    reply = segment("f")
    assert reply["ok"] and reply["unknown"] == ["nope"]
    assert [m["label"] for m in reply["masks"]] == ["vessel", "disc"]
    assert mask_of(reply["masks"][0]).sum() == H * W
    assert reply["masks"][0]["binary"] is True


def test_stack_result_maps_to_labels_in_order():
    @register_seg
    def f(image):
        out = np.zeros((2, H, W), np.float32)
        out[1, 0, 0] = 0.9
        out[1, 0, 1] = 0.2
        return out

    reply = segment("f")
    assert [m["label"] for m in reply["masks"]] == LABELS
    assert mask_of(reply["masks"][1]).tolist()[0][:2] == [1, 0]


def test_single_mask_targets_active_label_and_keeps_instance_ids():
    @register_seg
    def f(image):
        return np.full((H, W), 7, np.int64)

    (m,) = segment("f")["masks"]
    assert m["label"] is None and m["binary"] is False
    assert mask_of(m).max() == 7


def test_extras_are_passed_by_name():
    seen = {}

    @register_seg
    def f(image, masks, labels, active_label, frame_index):
        seen.update(masks=masks, labels=labels, active=active_label, index=frame_index)
        image[0, 0] = 1  # writable

    drawn = np.zeros((H, W), np.uint8)
    drawn[1, 1] = 3
    assert segment("f", masks={"disc": payload(drawn)}) == {"ok": True, "masks": [], "unknown": []}
    assert seen["labels"] == LABELS and seen["active"] == "disc" and seen["index"] == 2
    assert seen["masks"]["disc"][1, 1] == 3
    assert seen["masks"]["vessel"].shape == (H, W) and not seen["masks"]["vessel"].any()


@pytest.mark.parametrize("result, match", [
    (np.zeros((H + 1, W)), "shape"),
    (np.zeros((3, H, W)), "3 masks for 2 labels"),
])
def test_bad_results_report_an_error(result, match):
    @register_seg
    def f(image):
        return result

    reply = segment("f")
    assert not reply["ok"] and match in reply["error"]


def test_segment_refuses_other_kinds():
    @register_sequence_seg
    def track(frames):
        pass

    assert not segment("track")["ok"]


def test_sequence_single_stack_targets_active_label():
    @register_sequence_seg
    def track(frames, frame_index):
        assert frames.shape == (3, H, W, 3) and frame_index == 1
        return frames[..., 0] > 0  # T×H×W: frame 0 empty, others full

    run, results = run_sequence("track")
    assert run["ok"]
    assert [mask_of(r["masks"][0]).sum() for r in results] == [0, H * W, H * W]
    assert results[0]["masks"][0]["label"] is None


def test_sequence_dict_and_masks():
    @register_sequence_seg
    def track(frames, masks):
        assert masks["vessel"].shape == (3, H, W)
        return {"disc": masks["vessel"] * 2}

    drawn = np.ones((H, W), np.uint8)
    run, results = run_sequence("track", masks=[{"vessel": payload(drawn)}, {}, {}])
    assert run["ok"]
    assert [int(mask_of(r["masks"][0]).max()) for r in results] == [2, 0, 0]
    assert results[0]["masks"][0]["label"] == "disc"


def test_sequence_per_label_stack_and_length_check():
    @register_sequence_seg
    def per_label(frames):
        return np.ones((3, 2, H, W), bool)

    @register_sequence_seg
    def short(frames):
        return np.ones((2, H, W), bool)

    run, results = run_sequence("per_label")
    assert [m["label"] for m in results[2]["masks"]] == LABELS

    run, _ = run_sequence("short")
    assert not run["ok"] and "2 frames" in run["error"]


def test_arguments_are_described_in_ping():
    @register_seg("unet", threshold=Float(0.5, min=0, max=1, step=0.05, label="Cut-off"),
                  iterations=Int(3, min=1), clean=Boolean(True, hint="Removes specks"),
                  mode=Enum(["fast", "accurate"], default="accurate"))
    def f(image, masks, threshold, iterations, clean, mode):
        pass

    (fn,) = call({"op": "ping"})["functions"]
    assert fn["wants"] == ["masks"]
    assert fn["params"] == [
        {"name": "threshold", "type": "float", "label": "Cut-off", "default": 0.5,
         "hint": "Number from 0 to 1", "min": 0.0, "max": 1.0, "step": 0.05},
        {"name": "iterations", "type": "int", "label": "Iterations", "default": 3,
         "hint": "Integer, at least 1", "min": 1.0},
        {"name": "clean", "type": "bool", "label": "Clean", "default": True,
         "hint": "Removes specks"},
        {"name": "mode", "type": "enum", "label": "Mode", "default": "accurate", "hint": "",
         "choices": ["fast", "accurate"]},
    ]


def test_arguments_reach_the_function_with_defaults():
    seen = {}

    @register_seg(threshold=Float(0.5), n=Int(2), flag=Boolean(), mode=Enum(["a", "b"]))
    def f(image, active_label, threshold, n, flag, mode):
        seen.update(active=active_label, threshold=threshold, n=n, flag=flag, mode=mode)

    assert segment("f", args={"threshold": 1, "n": 4.0, "mode": "b"})["ok"]
    assert seen == {"active": "disc", "threshold": 1.0, "n": 4, "flag": False, "mode": "b"}
    assert isinstance(seen["threshold"], float) and isinstance(seen["n"], int)

    # An older Didascalie sends no arguments.
    assert segment("f")["ok"]
    assert (seen["threshold"], seen["n"], seen["mode"]) == (0.5, 2, "a")


@pytest.mark.parametrize("args, match", [
    ({"t": 2.0}, "t: 2.0 is above"),
    ({"n": 1.5}, "n: expected an integer"),
    ({"n": True}, "n: expected an integer"),
    ({"b": 1}, "b: expected true or false"),
    ({"e": "z"}, "e: 'z' is not one of"),
    ({"other": 1}, "Unknown arguments: other"),
])
def test_bad_arguments_are_refused(args, match):
    called = []

    @register_seg(t=Float(0.5, max=1), n=Int(1), b=Boolean(), e=Enum(["x"]))
    def f(image, t, n, b, e):
        called.append(1)

    reply = segment("f", args=args)
    assert not reply["ok"] and match in reply["error"] and not called


def test_sequence_arguments_are_checked_at_begin():
    seen = {}

    @register_sequence_seg("track", radius=Int(1, min=0))
    def track(frames, radius):
        seen["radius"] = radius

    begin = {"op": "seq_begin", "name": "track", "n_frames": 1, "labels": LABELS,
             "active_label": None, "frame_index": 0}
    assert not call({**begin, "args": {"radius": -1}})["ok"]
    assert call({**begin, "args": {"radius": 5}})["ok"]
    assert call({"op": "seq_frame", "index": 0,
                 "image": payload(np.zeros((H, W, 3), np.uint8))})["ok"]
    assert call({"op": "seq_run"})["ok"] and seen["radius"] == 5


def test_a_parameter_can_be_called_name():
    @register_seg("f", name=Enum(["x"]))
    def f(image, name):
        return None

    assert segment("f")["ok"]


@pytest.mark.parametrize("declared, match", [
    ({"missing": Int(1)}, "not a parameter"),
    ({"masks": Int(1)}, "reserved"),
    ({"image": Int(1)}, "image parameter"),
    ({"x": 3}, "must be an Int"),
])
def test_bad_declarations_fail_at_decoration(declared, match):
    with pytest.raises(TypeError, match=match):
        @register_seg("f", **declared)
        def f(image, masks, x=0):
            pass


@pytest.mark.parametrize("build", [
    lambda: Float(2, max=1),
    lambda: Int(1.5),
    lambda: Int(1, min=3, max=2),
    lambda: Float(1, step=0),
    lambda: Enum([]),
    lambda: Enum(["a"], default="b"),
    lambda: Boolean(1),
])
def test_bad_types_fail_when_built(build):
    with pytest.raises(ValueError):
        build()


def test_frame_name_is_passed_when_declared():
    seen = {}

    @register_seg
    def f(image, frame_name):
        seen["name"] = frame_name

    assert segment("f", frame_name="sub/a.png")["ok"] and seen["name"] == "sub/a.png"
    # An older Didascalie does not send it.
    assert segment("f")["ok"] and seen["name"] is None


def test_result_carries_a_message_with_or_without_masks():
    @register_seg
    def note(image):
        return Result(message="Saved 3 fibers")

    @register_seg
    def both(image):
        return Result({"vessel": np.ones((H, W), bool)}, message="Done")

    assert segment("note") == {"ok": True, "masks": [], "unknown": [],
                               "message": "Saved 3 fibers"}
    reply = segment("both")
    assert reply["message"] == "Done" and [m["label"] for m in reply["masks"]] == ["vessel"]


def test_sequence_result_carries_a_message():
    @register_sequence_seg
    def track(frames, frame_name):
        return Result(frames[..., 0] > 0, message=f"From {frame_name}")

    assert call({"op": "seq_begin", "name": "track", "n_frames": 1, "labels": LABELS,
                 "active_label": "disc", "frame_index": 0, "frame_name": "a.png"})["ok"]
    assert call({"op": "seq_frame", "index": 0,
                 "image": payload(np.ones((H, W, 3), np.uint8))})["ok"]
    run = call({"op": "seq_run"})
    assert run["ok"] and run["message"] == "From a.png"
    assert mask_of(call({"op": "seq_result", "index": 0})["masks"][0]).sum() == H * W

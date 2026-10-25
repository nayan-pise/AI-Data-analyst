import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyst.data_store import DataStore  # noqa: E402
from generate_sample_data import main as generate  # noqa: E402


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory) -> Path:
    return generate(tmp_path_factory.mktemp("sample"))


@pytest.fixture()
def store(sample_dir) -> DataStore:
    s = DataStore()
    for name in ("orders", "customers", "products"):
        s.load_csv(sample_dir / f"{name}.csv")
    return s


# ---- scripted fake of the Anthropic streaming client -------------------------------------------
class FakeStream:
    def __init__(self, message, texts):
        self.message, self.texts = message, texts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        yield from self.texts

    def get_final_message(self):
        return self.message


class FakeClient:
    def __init__(self, scripted):
        self.scripted, self.calls, self.messages = list(scripted), [], self

    def stream(self, **kwargs):
        self.calls.append(copy.deepcopy({k: v for k, v in kwargs.items() if k != "tools"}))
        if not self.scripted:
            raise RuntimeError("no more scripted responses")
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeStream(*item)


def text_msg(text, tin=10, tout=5):
    return (SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                            usage=SimpleNamespace(input_tokens=tin, output_tokens=tout), stop_reason="end_turn"), [text])


def tool_msg(name, args, tool_id="tu_1", preface="Let me check.", tin=20, tout=8):
    blocks = [SimpleNamespace(type="text", text=preface),
              SimpleNamespace(type="tool_use", id=tool_id, name=name, input=args)]
    return (SimpleNamespace(content=blocks, usage=SimpleNamespace(input_tokens=tin, output_tokens=tout),
                            stop_reason="tool_use"), [preface])

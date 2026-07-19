from beat_this import cli


class FakeFile2File:
    failures = set()

    def __init__(self, *args):
        pass

    def __call__(self, input_path, output_path):
        if input_path.name in self.failures:
            raise RuntimeError("decode failed")
        output_path.write_text("ok", encoding="utf-8")


def run_batch(input_dir, output_dir):
    return cli.run(
        inputs=[str(input_dir)],
        model="final0",
        output=str(output_dir),
        suffix=".beats",
        append=False,
        skip_existing=True,
        touch_first=True,
        dbn=False,
        gpu=-1,
        float16=False,
        activations=False,
    )


def test_failed_batch_item_is_reported_and_can_be_retried(
    monkeypatch, tmp_path, capsys
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "good.wav").touch()
    (input_dir / "bad.wav").touch()
    monkeypatch.setattr(cli, "File2File", FakeFile2File)

    FakeFile2File.failures = {"bad.wav"}
    assert run_batch(input_dir, output_dir) == 1
    assert (output_dir / "good.beats").read_text(encoding="utf-8") == "ok"
    assert not (output_dir / "bad.beats").exists()
    error_output = capsys.readouterr().err
    assert "Could not process" in error_output
    assert "bad.wav" in error_output

    FakeFile2File.failures = set()
    assert run_batch(input_dir, output_dir) == 0
    assert (output_dir / "bad.beats").read_text(encoding="utf-8") == "ok"

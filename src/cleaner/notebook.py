from __future__ import annotations

from pathlib import Path

from PIL import Image

from cleaner.config import AppConfig
from cleaner.labeling import record_action
from cleaner.utils import scan_photos


def start_labeler(
    photo_dir: str | Path,
    cfg: AppConfig,
    limit: int = 100,
    start_index: int = 0,
) -> None:
    """Launch a small ipywidgets labeler for Colab/Jupyter.

    This intentionally stays outside the core package imports so normal scripts
    do not require IPython or ipywidgets.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as exc:
        raise RuntimeError("Install ipywidgets to use the notebook labeler.") from exc

    paths = scan_photos(photo_dir)
    if limit > 0:
        paths = paths[:limit]
    state = {"index": max(start_index, 0)}

    buttons = {
        "discard": widgets.Button(description="Discard", button_style="danger"),
        "keep": widgets.Button(description="Keep", button_style="success"),
        "important": widgets.Button(description="Important", button_style="warning"),
        "later": widgets.Button(description="Later"),
    }
    output = widgets.Output()

    def render() -> None:
        with output:
            clear_output(wait=True)
            index = state["index"]
            if index >= len(paths):
                print("Done. Labels saved to:", cfg.labels_csv)
                return

            path = paths[index]
            print(f"{index + 1}/{len(paths)}")
            print(path)
            image = Image.open(path)
            image.thumbnail((700, 700))
            display(image)
            display(widgets.HBox(list(buttons.values())))

    def handle(action: str):
        def _on_click(_button) -> None:
            index = state["index"]
            if index < len(paths):
                record_action(cfg, paths[index], action=action, source="notebook_labeler")
                state["index"] += 1
            render()

        return _on_click

    for action, button in buttons.items():
        button.on_click(handle(action))

    display(output)
    render()

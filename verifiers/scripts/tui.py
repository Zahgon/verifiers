"""
Textual-based TUI for viewing verifiers eval results.
"""

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, cast

from markdown_it import MarkdownIt
from mdit_py_plugins.amsmath import amsmath_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from rich import box
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text
from textual import events, on, work
from textual.dom import DOMNode
from textual.widget import Widget
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.content import Content, Span
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.style import Style
from textual.theme import Theme
from textual.widgets import (
    Collapsible,
    Footer,
    Input,
    Label,
    OptionList,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
    Tree,
)
from textual.widgets._markdown import (
    Markdown as BaseMarkdown,
    MarkdownBlock,
    MarkdownH1,
    MarkdownH2,
    MarkdownH3,
    MarkdownH4,
    MarkdownH5,
    MarkdownH6,
    MarkdownParagraph,
    MarkdownTD,
    MarkdownTH,
)
from textual.widgets._option_list import Option
from textual.widgets._tabbed_content import ContentTabs
from textual.widgets._tree import TreeNode

from verifiers.utils.display_utils import format_numeric

AnimationLevel = Literal["none", "basic", "full"]
TreeBinding = Binding | tuple[str, str] | tuple[str, str, str]


def _binding_key(binding: TreeBinding) -> str:
    if isinstance(binding, Binding):
        return binding.key
    return binding[0]


def _int_like_sort_key(value: Any) -> Tuple[int, int, str]:
    pass


# ----------------------------
# Discovery and data loading
# ----------------------------
@dataclass
class RunInfo:
    env_id: str
    model: str
    run_id: str
    path: Path
    metadata: Optional[Dict[str, Any]] = None

    def load_metadata(self) -> Dict[str, Any]:
        pass


@dataclass(frozen=True)
class BrowserNodeData:
    kind: str
    env_id: str = ""
    model: str = ""
    run: Optional[RunInfo] = None
    tree_name: str = ""
    tree_suffix: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class MetricSummary:
    name: str
    count: int
    avg: float
    min_value: float
    max_value: float


@dataclass(frozen=True)
class RunOverviewStats:
    rewards: List[float]
    metric_summaries: List[MetricSummary]


class RunBrowserTree(Tree[BrowserNodeData]):
    """Tree with footer-visible shortcuts for the eval browser."""

    BINDINGS = [
        *(
            binding
            for binding in Tree.BINDINGS
            if _binding_key(binding) not in {"enter", "space"}
        ),
        Binding("left", "cursor_parent", "Parent folder", show=True),
        Binding("right", "cursor_right", "Expand/next folder", show=True),
        Binding("enter", "select_cursor", "Open/toggle", show=True),
        Binding("space", "toggle_node", "Toggle folder", show=True),
    ]

    def _visible_depth(self, node: Any) -> int:
        pass

    def _render_browser_label(
        self, payload: BrowserNodeData, style: Style, max_width: int
    ) -> Text:
        pass

    def render_label(  # ty: ignore[invalid-method-override]
        self,
        node: TreeNode[Any],
        base_style: Style,
        style: Style,
    ) -> Text:
        pass

    def action_cursor_parent(self) -> None:
        """Move the cursor to the nearest visible parent folder."""
        pass

    def action_cursor_right(self) -> None:
        """Expand the current folder or move to the next visible parent folder."""
        pass

    def action_toggle_node(self) -> None:
        """Toggle the current folder, or the nearest ancestor folder for a leaf."""
        pass


def discover_results(
    env_dir_path: str = "./environments", outputs_dir_path: str = "./outputs"
) -> Dict[str, Dict[str, List[RunInfo]]]:
    """
    Returns mapping: env_id -> model -> list[RunInfo]
    """
    roots: List[Path] = []
    env_dir = Path(env_dir_path)
    if env_dir.is_dir():
        for env_path in sorted(env_dir.iterdir(), key=lambda path: path.name):
            candidate = env_path / "outputs" / "evals"
            if candidate.is_dir():
                roots.append(candidate)

    global_root = Path(outputs_dir_path) / "evals"
    if global_root.is_dir():
        roots.append(global_root)

    discovered: Dict[str, Dict[str, List[RunInfo]]] = {}
    for root in roots:
        for env_model_dir in sorted(root.iterdir(), key=lambda path: path.name):
            if not env_model_dir.is_dir() or "--" not in env_model_dir.name:
                continue
            env_id, model_part = env_model_dir.name.split("--", 1)
            model = model_part.replace("--", "/")
            for run_dir in sorted(env_model_dir.iterdir(), key=lambda path: path.name):
                if not run_dir.is_dir():
                    continue
                if (run_dir / "metadata.json").is_file() and (
                    run_dir / "results.jsonl"
                ).is_file():
                    run = RunInfo(
                        env_id=env_id,
                        model=model,
                        run_id=run_dir.name,
                        path=run_dir,
                    )
                    discovered.setdefault(env_id, {}).setdefault(model, []).append(run)

    return discovered


class LazyRunResults:
    """Lazy loader for results.jsonl with optional metadata count."""

    def __init__(self, run: RunInfo):
        self._path = run.path / "results.jsonl"
        self._fh = self._path.open("r", encoding="utf-8")
        self._offsets: List[int] = []
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._eof = False
        self._count_hint: Optional[int] = None
        self._count: Optional[int] = None

        meta = run.load_metadata()
        num_examples = meta.get("num_examples")
        rollouts_per_example = meta.get("rollouts_per_example")
        if isinstance(num_examples, int) and num_examples >= 0:
            if isinstance(rollouts_per_example, int) and rollouts_per_example >= 0:
                self._count_hint = num_examples * rollouts_per_example
            else:
                self._count_hint = num_examples

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def _read_next_line(self) -> Optional[str]:
        if self._eof:
            return None
        pos = self._fh.tell()
        line = self._fh.readline()
        if not line:
            self._eof = True
            self._count = len(self._offsets)
            return None
        self._offsets.append(pos)
        return line

    def _ensure_index(self, index: int) -> bool:
        if index < 0:
            return False
        while len(self._offsets) <= index and not self._eof:
            line = self._read_next_line()
            if line is None:
                break
        return index < len(self._offsets)

    def _ensure_count(self) -> int:
        pass

    def get(self, index: int) -> Dict[str, Any]:
        if index in self._cache:
            return self._cache[index]
        if not self._ensure_index(index):
            return {}
        pos = self._fh.tell()
        try:
            self._fh.seek(self._offsets[index])
            line = self._fh.readline()
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                data = {}
        finally:
            self._fh.seek(pos)
        self._cache[index] = data
        return data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(index)

    def __len__(self) -> int:
        return self._ensure_count()

    def __bool__(self) -> bool:
        if self._count is not None:
            return self._count > 0
        if self._offsets:
            return True
        if self._eof:
            return False
        line = self._read_next_line()
        return line is not None

    def count_hint(self) -> Optional[int]:
        pass


class LazyLogFile:
    """Lazy loader for log files with line-level random access."""

    MAX_DISPLAY_LINES = 10_000

    def __init__(self, path: Path):
        self._path = path
        self._fh = path.open("r", encoding="utf-8", errors="replace")
        self._offsets: List[int] = []
        self._cache: Dict[int, str] = {}
        self._eof = False
        self._count: Optional[int] = None

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def _read_next_line(self) -> Optional[str]:
        if self._eof:
            return None
        pos = self._fh.tell()
        line = self._fh.readline()
        if not line:
            self._eof = True
            self._count = len(self._offsets)
            return None
        self._offsets.append(pos)
        return line

    def _ensure_index(self, index: int) -> bool:
        if index < 0:
            return False
        while len(self._offsets) <= index and not self._eof:
            if self._read_next_line() is None:
                break
        return index < len(self._offsets)

    def _ensure_count(self) -> int:
        pass

    def get_line(self, index: int) -> str:
        pass

    def __len__(self) -> int:
        return self._ensure_count()

    def __bool__(self) -> bool:
        if self._count is not None:
            return self._count > 0
        if self._offsets:
            return True
        if self._eof:
            return False
        return self._read_next_line() is not None


# ----------------------------
# Log styling helpers
# ----------------------------

_LOG_LEVEL_STYLES: Dict[str, str] = {
    "DEBUG": "dim blue",
    "INFO": "bold green",
    "WARNING": "bold yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold red reverse",
}


def _parse_log_header(line: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse a log line into (timestamp, source, level, message).

    Expected format: '2026-03-03 22:57:21 - source.name - LEVEL ...'
    """
    if len(line) < 22 or line[19:22] != " - ":
        return None
    rest = line[22:]
    sep_idx = rest.find(" - ")
    if sep_idx < 0:
        return None
    source = rest[:sep_idx]
    after_source = rest[sep_idx + 3 :]
    space_idx = after_source.find(" ")
    if space_idx < 0:
        level = after_source
        message = ""
    else:
        level = after_source[:space_idx]
        message = after_source[space_idx:]
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        return None
    return line[:19], source, level, message


def _append_styled_log_line(log_text: Text, line: str) -> None:
    """Append a log line to a Text object with colored header parts."""
    parsed = _parse_log_header(line)
    if parsed is None:
        log_text.append(line, style="dim")
        return
    timestamp, source, level, message = parsed
    level_style = _LOG_LEVEL_STYLES.get(level, "dim")
    log_text.append(timestamp, style="bold dim")
    log_text.append(" - ", style="dim")
    log_text.append(source, style="dim cyan")
    log_text.append(" - ", style="dim")
    log_text.append(level, style=level_style)
    log_text.append(message, style="dim")


def _log_tab_label(path: Path) -> str:
    """Derive a display label from a log file path."""
    pass


def _discover_log_files(run_path: Path) -> List[Path]:
    """Find log files in a run directory, sorted with env_server first."""
    pass


def _merge_log_files(log_files: List[Path]) -> List[str]:
    """Merge lines from multiple log files, sorted by timestamp.

    Lines without a parseable timestamp are attached to the preceding
    timestamped line (continuation lines from multi-line log messages).
    """
    pass


# ----------------------------
# Formatting helpers
# ----------------------------


def _stringify_message_content(content: Any) -> str:
    """Render message content into readable plain text."""
    pass


def _thinking_block_to_text(block: Any) -> str:
    pass


def _stringify_message_reasoning(message: Any) -> str:
    pass


def _stringify_message(message: Any) -> str:
    pass


def _parse_tool_calls(tool_calls: Any) -> List[Any]:
    pass


def _truncate_preview(text: str, limit: int = 72) -> str:
    pass


def _compute_prompt_hash(prompt: list | None) -> str | None:
    """MD5 hash of JSON-serialized prompt for deduplication."""
    pass


def _compute_run_overview_stats(run: RunInfo) -> RunOverviewStats:
    pass


def _format_message_preview(message: Any) -> str:
    pass


def _reward_style(value: Any) -> str:
    pass


def _format_reward_value(value: Any) -> str:
    pass


def _format_compact_metric(value: Any) -> str:
    pass


def _numeric_reward(value: Any) -> Optional[float]:
    pass


def _pretty_json_or_str(value: Any) -> str:
    pass


def _compact_json_or_str(value: Any) -> str:
    pass


def _format_setting_value(value: Any) -> str:
    pass


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return str(getattr(tool, "name", "") or "")
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str):
            return name
    name = tool.get("name")
    return name if isinstance(name, str) else ""


def _run_setting_rows(meta: Dict[str, Any]) -> List[Tuple[str, str]]:
    pass


def _build_settings_table(
    rows: List[Tuple[str, str]],
    heading: str,
    *,
    value_header: str = "Value",
) -> Group | Text:
    pass


def _run_setting_variation_rows(
    runs: List[RunInfo], *, max_rows: int = 8
) -> Tuple[List[Tuple[str, str]], int]:
    pass


def _varying_run_setting_keys(
    runs: List[RunInfo],
) -> Tuple[List[str], List[Tuple[RunInfo, Dict[str, str]]]]:
    pass


def _reward_bucket_counts(values: List[float]) -> List[Tuple[str, int, str]]:
    pass


_COMPARE_ALIAS_PALETTE: Tuple[str, ...] = (
    "#61afef",
    "#98c379",
    "#e5c07b",
    "#c678dd",
    "#56b6c2",
    "#e06c75",
)


def _tool_call_parts(tool_call: Any) -> Tuple[str, str, Optional[str]]:
    pass


def _tool_output_preview(message: Any) -> str:
    pass


def _tool_group_preview(message: Any, tool_outputs: List[Any]) -> str:
    pass


def _raw_preview(value: Any, *, limit: int = 56) -> str:
    pass


def _error_preview(error: Any) -> str:
    pass


def _parse_jsonish_string(value: Any) -> Any:
    pass


def format_info_for_details(info: Any) -> str:
    """Format record info for the details panel in rollout view."""
    pass


_STANDARD_NUMERIC_FIELDS = {
    "example_id",
    "prompt",
    "completion",
    "answer",
    "task",
    "info",
    "reward",
    "error",
    "timing",
    "is_completed",
    "is_truncated",
    "stop_condition",
    "metrics",
    "tool_defs",
    "token_usage",
    "error_chain",
    "long_error_chain",
}


def _extract_numeric_metric_values(record: Dict[str, Any]) -> Dict[str, float]:
    pass


def _build_reward_distribution_table(values: List[float], heading: str) -> Group | Text:
    pass


def _format_metric_stat_value(value: float) -> str:
    pass


def _build_metric_summary_table(metric_summaries: List[MetricSummary]) -> Table | Text:
    pass


# ----------------------------
# Custom Panel Widget
# ----------------------------
class Panel(Container):
    """A rounded panel container."""

    pass


class TabbedScrollPane(VerticalScroll):
    """A VerticalScroll that switches sibling tabs with left/right arrows."""

    BINDINGS = [
        Binding("left", "prev_tab", "Prev tab", show=False),
        Binding("right", "next_tab", "Next tab", show=False),
    ]

    def _get_tabbed_content(self) -> TabbedContent | None:
        pass

    def action_prev_tab(self) -> None:
        pass

    def action_next_tab(self) -> None:
        pass


class LogScrollPane(VerticalScroll):
    """A VerticalScroll that switches log file tabs with left/right arrows."""

    BINDINGS = [
        Binding("left", "prev_log_tab", "Prev log", show=False),
        Binding("right", "next_log_tab", "Next log", show=False),
    ]

    def _get_view_run_screen(self) -> Optional["ViewRunScreen"]:
        pass

    def action_prev_log_tab(self) -> None:
        pass

    def action_next_log_tab(self) -> None:
        pass


# ----------------------------
# Search helpers
# ----------------------------
@dataclass(frozen=True)
class SearchHit:
    column: str
    line_index: int
    line_text: str
    section_index: int = 0
    nested_index: int = -1  # -1 = parent body, 0+ = nested section index


@dataclass(frozen=True)
class SearchResult:
    column: str
    pattern: str
    section_index: int = 0
    nested_index: int = -1


@dataclass(frozen=True)
class HistorySectionData:
    title: str
    body: str
    column: str
    collapsed: bool
    classes: str
    nested_sections: Tuple["HistorySectionData", ...] = ()
    body_first: bool = True


@dataclass(frozen=True)
class RolloutCopyItem:
    key: str
    label: str
    body: str


def _stylize_matches(text: Text, pattern: re.Pattern, style: str) -> Text:
    pass


def _sorted_runs(runs: List[RunInfo]) -> List[RunInfo]:
    pass


def _format_run_datetime(meta: Dict[str, Any]) -> str:
    pass


def _text_to_plain(text: Text) -> str:
    pass


def _indent_block(text: str, prefix: str) -> str:
    pass


# ----------------------------
# Markdown rendering
# ----------------------------
_LATEX_BEGIN_END_RE = re.compile(r"\\(?:begin|end)\{[^}]+\}")
_LATEX_BRACED_SCRIPT_RE = re.compile(r"([_^])\{([^{}]+)\}")
_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|operatorname|text)\{([^{}]+)\}"
)
_LATEX_FRACTION_RE = re.compile(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\{([^{}]+)\}")
_LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+|.)")
_LATEX_COMMAND_REPLACEMENTS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "sigma": "σ",
    "phi": "φ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "cdot": "·",
    "times": "×",
    "pm": "±",
    "neq": "!=",
    "leq": "<=",
    "geq": ">=",
    "approx": "~",
    "to": "->",
    "rightarrow": "->",
    "leftarrow": "<-",
    "infty": "∞",
    "ldots": "...",
    "cdots": "...",
    "sum": "sum",
    "prod": "prod",
    "log": "log",
    "ln": "ln",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "|": "||",
    ",": " ",
    ";": " ",
    "!": "",
}


def _replace_latex_groups(
    text: str,
    pattern: re.Pattern[str],
    replacement: str | Callable[[re.Match[str]], str],
) -> str:
    pass


def _replace_latex_fraction(match: re.Match[str]) -> str:
    pass


def _replace_latex_command(match: re.Match[str]) -> str:
    pass


def _fallback_latex_to_text(latex: str, *, preserve_newlines: bool) -> str:
    pass


def _latex_to_text(latex: str, *, preserve_newlines: bool) -> str:
    pass


def render_inline_math(latex: str) -> str:
    pass


def render_block_math(latex: str) -> str:
    pass


def make_math_parser() -> MarkdownIt:
    pass


class MathInlineMixin:
    """Teach Textual's Markdown blocks how to render inline math tokens."""

    def _token_to_content(self, token: Any) -> Content:
        pass


class MathParagraph(MathInlineMixin, MarkdownParagraph):
    pass


class MathH1(MathInlineMixin, MarkdownH1):
    pass


class MathH2(MathInlineMixin, MarkdownH2):
    pass


class MathH3(MathInlineMixin, MarkdownH3):
    pass


class MathH4(MathInlineMixin, MarkdownH4):
    pass


class MathH5(MathInlineMixin, MarkdownH5):
    pass


class MathH6(MathInlineMixin, MarkdownH6):
    pass


class MathTH(MathInlineMixin, MarkdownTH):
    pass


class MathTD(MathInlineMixin, MarkdownTD):
    pass


class MathDisplayBlock(MarkdownBlock):
    DEFAULT_CSS = """
    MathDisplayBlock {
        width: 1fr;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        background: $boost;
        border-left: outer $primary 60%;
    }
    """

    def __init__(self, markdown: "MathMarkdown", token: Any):
        super().__init__(markdown, token)
        text = render_block_math(token.content)
        if token.type == "math_block_label" and getattr(token, "info", ""):
            text = f"[{token.info}]\n{text}"
        self.set_content(Content(text))


class MathMarkdown(BaseMarkdown):
    BLOCKS = BaseMarkdown.BLOCKS | {
        "paragraph_open": MathParagraph,
        "h1": MathH1,
        "h2": MathH2,
        "h3": MathH3,
        "h4": MathH4,
        "h5": MathH5,
        "h6": MathH6,
        "th_open": MathTH,
        "td_open": MathTD,
    }

    def __init__(self, markdown: str | None = None, **kwargs: Any) -> None:
        super().__init__(markdown, parser_factory=make_math_parser, **kwargs)

    def unhandled_token(self, token: Any) -> MarkdownBlock | None:
        pass


# ----------------------------
# Screens
# ----------------------------
class CompareRunsScreen(Screen):
    """Dedicated comparison view for runs, optionally across models."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b,backspace", "back", "Back"),
        Binding("g", "enter_group_mode", "Group by"),
        Binding("left", "group_cursor_left", show=False),
        Binding("right", "group_cursor_right", show=False),
        Binding("enter", "group_select", show=False),
        Binding("escape", "exit_group_mode", show=False),
        Binding("c", "copy", "Copy"),
        Binding("ctrl+c", "copy", show=False),
    ]

    def __init__(self, env_id: str, model: Optional[str], runs: List[RunInfo]):
        super().__init__()
        self.env_id = env_id
        self.model = model
        self.runs = list(runs)
        self._stats_by_path: Dict[Path, RunOverviewStats] = {}
        self._setting_keys: List[str] = []
        self._run_settings: List[Tuple[RunInfo, Dict[str, str]]] = []
        self._group_mode: bool = False
        self._group_cursor: int = 0
        self._grouped_by_key: str | None = None
        self._distinct_prompts_by_group: Dict[Tuple[str, ...], int] = {}
        self._prompt_count_cache: Dict[str, int] = {}  # run-ID-set hash → count

    def compose(self) -> ComposeResult:
        pass

    def on_mount(self) -> None:
        pass

    def action_back(self) -> None:
        pass

    @staticmethod
    def _renderable_to_text(renderable: Any, width: int = 220) -> str:
        pass

    def action_copy(self) -> None:
        pass

    @work(
        thread=True,
        group="run-comparison",
        exclusive=True,
        exit_on_error=False,
    )
    def _load_comparison_stats(self) -> None:
        pass

    def _finish_loading_comparison_stats(
        self, stats_by_path: Dict[Path, RunOverviewStats]
    ) -> None:
        pass

    def on_resize(self, event: events.Resize) -> None:
        pass

    def _refresh_outcomes(self) -> None:
        pass

    @work(
        thread=True,
        group="prompt-counting",
        exclusive=True,
        exit_on_error=False,
    )
    def _load_distinct_prompt_counts(self) -> None:
        """Compute distinct prompt counts per group in a background thread."""
        pass

    def _update_group_prompt_count(
        self, group_key: Tuple[str, ...], count: int
    ) -> None:
        pass

    def action_enter_group_mode(self) -> None:
        pass

    def action_group_cursor_left(self) -> None:
        pass

    def action_group_cursor_right(self) -> None:
        pass

    def action_group_select(self) -> None:
        pass

    def action_exit_group_mode(self) -> None:
        pass

    def _short_setting_key(self, key: str) -> str:
        pass

    def _alias_style(self, label: str) -> str:
        pass

    def _share_style(self, share: float, positive: bool) -> str:
        pass

    def _build_reward_mix_bar(self, values: List[float], width: int = 18) -> Text:
        pass

    def _build_grouped_outcomes_table(
        self,
        stats_by_path: Dict[Path, RunOverviewStats],
        setting_keys: List[str],
        run_settings: List[Tuple[RunInfo, Dict[str, str]]],
        group_by_key: str | None = None,
        highlight_col: int | None = None,
    ) -> Tuple[Table, List[Tuple[str, str, str]], List[Tuple[str, str, str, str]]]:
        # Determine which keys to actually group by.
        pass

    def _build_argument_legend(
        self,
        axis_rows: List[Tuple[str, str, str]],
        value_rows: List[Tuple[str, str, str, str]],
    ) -> Group | Text:
        """Build the argument legend.

        axis_rows: (alias, full_name, style)
        value_rows: (alias, full_name, preview, style)
        """
        pass

    def _build_comparison_outcomes(self) -> Group:
        pass


class BrowseRunsScreen(Screen):
    """Single-screen browser for environments, models, and runs."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("enter", "enter_selected", "Open/toggle", priority=True),
        Binding("tab", "focus_next_pane", "Next pane"),
        Binding("shift+tab", "focus_prev_pane", show=False),
        Binding("v", "compare_selected", "Compare"),
        Binding("c", "copy", "Copy"),
        Binding("ctrl+c", "copy", show=False),
    ]

    def __init__(self, index: Dict[str, Dict[str, List[RunInfo]]]):
        super().__init__()
        self.index = index
        self._run_overview_cache: Dict[Path, RunOverviewStats] = {}
        self._click_selected_node: object | None = None

    def compose(self) -> ComposeResult:
        pass

    def on_mount(self) -> None:
        pass

    def action_focus_next_pane(self) -> None:
        pass

    def action_focus_prev_pane(self) -> None:
        pass

    def action_copy(self) -> None:
        pass

    def action_compare_selected(self) -> None:
        pass

    def _populate_tree(self, tree: Tree) -> Any:
        pass

    @on(Tree.NodeHighlighted, "#run-browser-tree")
    def on_tree_highlighted(self, event: Tree.NodeHighlighted) -> None:
        pass

    def action_enter_selected(self) -> None:
        """Enter key: immediately open the highlighted run or toggle folder."""
        pass

    @on(Tree.NodeSelected, "#run-browser-tree")
    def on_tree_selected(self, event: Tree.NodeSelected) -> None:
        """Click: first click selects, second click enters rollout view."""
        pass

    def _details_for(self, payload: Any) -> Any:
        pass

    @work(
        thread=True,
        group="run-overview",
        exclusive=True,
        exit_on_error=False,
    )
    def _load_run_overview_stats(self, run: RunInfo) -> None:
        pass

    def _finish_loading_run_overview_stats(
        self, run: RunInfo, stats: RunOverviewStats
    ) -> None:
        pass

    def _build_env_details(self, env_id: str) -> Group:
        pass

    def _build_model_details(self, env_id: str, model: str) -> Group:
        pass

    def _build_run_details(
        self,
        run: RunInfo,
        stats: Optional[RunOverviewStats] = None,
    ) -> Group:
        pass


class ViewRunScreen(Screen):
    """Screen for viewing run details and rollouts."""

    COMPACT_LAYOUT_WIDTH = 150

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b,backspace", "back", "Back"),
        Binding("p", "prev_record", "Prev rollout"),
        Binding("n", "next_record", "Next rollout"),
        Binding("l", "show_logs", "Logs"),
        Binding("r", "show_rollouts", "Rollouts"),
        Binding("pageup", "history_page_up", show=False),
        Binding("pagedown", "history_page_down", show=False),
        Binding("home", "history_home", show=False),
        Binding("end", "history_end", show=False),
        Binding("tab", "focus_next_pane", "Next pane"),
        Binding("shift+tab", "focus_prev_pane", show=False),
        Binding("e", "expand_all", "Expand all"),
        Binding("x", "collapse_all", "Collapse all"),
        Binding("s", "search", "Search"),
        Binding("m", "toggle_markdown_math", "Toggle markdown"),
        Binding("c", "copy", "Copy"),
        Binding("ctrl+c", "copy", show=False),
    ]

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        """Dynamically show/hide footer bindings based on view mode."""
        pass

    def __init__(self, run: RunInfo):
        super().__init__()
        self.run = run
        self.records = LazyRunResults(run)
        self._record_count = self.records.count_hint()
        self.current_record_idx = 0
        self._prompt_text: str = ""
        self._completion_text: str = ""
        self._highlight_regex: Optional[re.Pattern] = None
        self._highlight_column: Optional[str] = None
        self._highlight_timer = None
        self._previous_animation_level: Optional[AnimationLevel] = None
        self._render_markdown_math = True
        # Log viewer state
        # Tab 0 = "all" (merged), tab 1+ = individual files
        self._log_files: List[Path] = _discover_log_files(run.path)
        self._log_loaders: Dict[int, LazyLogFile] = {}
        self._merged_log_lines: Optional[List[str]] = None
        self._active_log_tab: int = 0
        self._view_mode: Literal["rollouts", "logs"] = "rollouts"
        self._log_highlight_regex: Optional[re.Pattern] = None
        self._log_highlight_timer = None
        if self.records:
            self._set_record_text_state(self.records[self.current_record_idx])

    def compose(self) -> ComposeResult:
        pass

    def _build_header_summary_text(self) -> Text:
        pass

    def _build_history_summary_text(self, record: Dict[str, Any]) -> Text:
        pass

    def _build_header_metric_text(self) -> Text:
        pass

    def _build_reward_text(
        self,
        record: Dict[str, Any],
        *,
        heading: str,
        multiline: bool,
        limit: Optional[int] = None,
    ) -> Text:
        pass

    def _build_header_reward_text(self, record: Dict[str, Any]) -> Text:
        pass

    def on_mount(self) -> None:
        pass

    def on_resize(self, event: events.Resize) -> None:
        pass

    def on_unmount(self) -> None:
        pass

    def _available_record_count(self) -> int:
        pass

    def _record_progress_label(self) -> str:
        pass

    def _hydrate_rollout_option(self, index: int) -> None:
        pass

    def _populate_rollout_list(self) -> None:
        pass

    def _build_rollout_prompt(
        self,
        idx: int,
        record: Optional[Dict[str, Any]] = None,
    ) -> Text:
        pass

    def _record_preview(self, record: Dict[str, Any]) -> str:
        pass

    def _format_prompt_or_completion(self, prompt_or_completion: Any) -> Text:
        pass

    def _set_record_text_state(self, record: Dict[str, Any]) -> None:
        pass

    def update_display(self, *, focus_history: bool = False) -> None:
        pass

    def action_back(self) -> None:
        pass

    def action_prev_record(self) -> None:
        pass

    def action_next_record(self) -> None:
        pass

    def _move_record_cursor(self, delta: int) -> None:
        pass

    # ------ Log viewer ------

    def action_show_logs(self) -> None:
        pass

    def action_show_rollouts(self) -> None:
        pass

    def _cycle_log_tab(self, delta: int) -> None:
        pass

    def _log_tab_count(self) -> int:
        """Number of log tabs: 'all' + individual files if 2+ files, else just 1."""
        pass

    def _build_log_tab_bar(self) -> Text:
        pass

    def _get_active_log_lines(self) -> Tuple[List[str], str]:
        """Return (lines, tab_label) for the active log tab."""
        pass

    def _populate_logs_view(self) -> None:
        pass

    def _build_search_lines(
        self, record: Dict[str, Any]
    ) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
        """Build tagged (section_index, nested_index, line) lists for search."""
        pass

    def action_search(self) -> None:
        pass

    def _search_logs(self) -> None:
        pass

    def _handle_log_search_result(self, result: Optional[SearchResult]) -> None:
        pass

    def _scroll_to_first_log_match(self) -> None:
        """Scroll the logs panel so the first matching line is visible."""
        pass

    def _clear_log_highlight(self) -> None:
        pass

    def action_copy(self) -> None:
        pass

    def _copy_logs(self) -> None:
        pass

    def action_expand_all(self) -> None:
        pass

    def action_collapse_all(self) -> None:
        pass

    @on(TabbedContent.TabActivated, "#details-tabs")
    def on_details_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Focus the scroll pane in the newly active details tab."""
        pass

    def _should_skip_focus(self, widget: Widget) -> bool:
        """Return True for widgets that should be skipped during tab cycling."""
        pass

    def action_focus_next_pane(self) -> None:
        pass

    def action_focus_prev_pane(self) -> None:
        pass

    def _center_scroll_target(self) -> VerticalScroll:
        pass

    def action_history_page_up(self) -> None:
        pass

    def action_history_page_down(self) -> None:
        pass

    def action_history_home(self) -> None:
        pass

    def action_history_end(self) -> None:
        pass

    def _make_body_widget(self, body: str, column: str) -> Widget:
        """Create the appropriate body widget based on render mode."""
        pass

    def _collect_section_bodies(
        self, sections: List[HistorySectionData]
    ) -> List[Tuple[str, str]]:
        """Flatten all section (body, column) pairs in DOM order."""
        pass

    def _swap_section_bodies(self) -> None:
        """Re-render all .section-body widgets in-place (preserves collapsed state)."""
        pass

    def action_toggle_markdown_math(self) -> None:
        pass

    def _handle_search_result(self, result: Optional[SearchResult]) -> None:
        pass

    def _set_highlight(
        self, result: Optional[SearchResult], *, repaint: bool = True
    ) -> None:
        pass

    def _build_rollout_summary_text(self, record: Dict[str, Any]) -> Text:
        pass

    def _update_responsive_layout(self, width: int) -> None:
        pass

    def _set_current_record(self, index: int, *, focus_history: bool = False) -> None:
        pass

    @on(OptionList.OptionHighlighted, "#rollout-list")
    def on_rollout_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        pass

    @on(OptionList.OptionSelected, "#rollout-list")
    def on_rollout_selected(self, event: OptionList.OptionSelected) -> None:
        pass

    def _reasoning_section_data(
        self,
        message: Dict[str, Any],
        *,
        collapsed: bool = True,
    ) -> Tuple[HistorySectionData, ...]:
        pass

    def _history_section_data(self, record: Dict[str, Any]) -> List[HistorySectionData]:
        pass

    def _completion_sections(self, record: Dict[str, Any]) -> List[Collapsible]:
        pass

    def _rebuild_completion_sections(
        self, record: Dict[str, Any], focus_history: bool = False
    ) -> None:
        pass

    def _expand_and_scroll_to_match(self, container: VerticalScroll) -> None:
        """Expand the target section (and nested subsection) and scroll to it."""
        pass

    def _scroll_to_section(self, section: Collapsible) -> None:
        pass

    def _detail_copy_sections(
        self, record: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        pass

    def _render_detail_copy_text(self, sections: List[Tuple[str, str, str]]) -> str:
        pass

    def _render_history_section_copy_text(
        self, section: HistorySectionData, *, depth: int = 0
    ) -> str:
        pass

    def _render_history_copy_text(self, sections: List[HistorySectionData]) -> str:
        pass

    def _append_history_copy_items(
        self,
        items: List[RolloutCopyItem],
        sections: List[HistorySectionData],
        *,
        depth: int = 0,
        prefix: str = "history",
    ) -> None:
        pass

    def _build_rollout_snapshot_text(
        self,
        record: Dict[str, Any],
        history_sections: List[HistorySectionData],
        detail_sections: List[Tuple[str, str, str]],
    ) -> str:
        pass

    def _build_rollout_copy_items(
        self, record: Dict[str, Any]
    ) -> List[RolloutCopyItem]:
        pass

    def _history_groups(self, completion: List[Any]) -> List[Dict[str, Any]]:
        pass

    def _section_matches_highlight(self, section: HistorySectionData) -> bool:
        pass

    def _make_section(self, section: HistorySectionData) -> Collapsible:
        pass

    def _focus_primary_content(self, *, prefer_expanded: bool = True) -> None:
        pass

    @on(Collapsible.Expanded)
    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        pass

    def _shift_expand_pulse(self, collapsible: Collapsible) -> None:
        pass

    def _clear_expand_pulse(self, collapsible: Collapsible) -> None:
        pass

    def _build_score_text(self, record: Dict[str, Any]) -> Text:
        pass

    def _build_task_text(self, record: Dict[str, Any]) -> Text:
        pass

    def _build_usage_text(self, record: Dict[str, Any]) -> Text:
        pass

    def _build_info_text(self, record: Dict[str, Any]) -> Text:
        pass

    def _append_context_section(self, out: Text, title: str, value: Any) -> None:
        pass


# ----------------------------
# Main App
# ----------------------------
class VerifiersTUI(App):
    """Textual-based TUI for viewing verifiers eval results."""

    # Custom dark theme with a modern color palette
    ENABLE_COMMAND_PALETTE = False  # Disable command palette for cleaner UI

    # Define custom dark theme
    BLACK_WARM_THEME = Theme(
        name="black-warm",
        primary="#d4a373",  # Warm tan/beige
        secondary="#808080",  # Gray
        accent="#c9ada7",  # Muted rose
        warning="#ffa500",  # Orange
        error="#ff6b6b",  # Soft red
        success="#98c379",  # Soft green
        background="#141414",
        surface="#141414",
        panel="#141414",
        foreground="#ffffff",
        dark=True,
    )

    # Define custom light theme with matching warm tones
    WHITE_WARM_THEME = Theme(
        name="white-warm",
        primary="#8b6f47",  # Darker warm brown (darker than dark theme for contrast)
        secondary="#606060",  # Medium gray
        accent="#a08b87",  # Muted warm brown-rose
        warning="#ff8c00",  # Dark orange
        error="#dc143c",  # Crimson
        success="#6b8e23",  # Olive green
        background="#f5f5f5",  # Light warm grey
        surface="#f5f5f5",  # Light warm grey
        panel="#f5f5f5",  # Light warm grey
        foreground="#1a1a1a",  # Near black
        dark=False,
    )

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
    ]

    CSS = """
    /* Clean black theme */
    Screen {
        layout: vertical;
        background: $background;
    }
    
    Panel {
        border: round $primary;
        padding: 1 2;
        margin: 0 0 1 0;
        background: $panel;
    }
    
    Label {
        color: $text;
    }
    
    Static {
        color: $text;
    }
    
    .title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    .subtitle {
        color: $text-muted;
        margin-bottom: 1;
    }

    .copy-hint {
        color: $text-muted;
        margin-bottom: 0;
    }

    
    OptionList {
        height: auto;
        max-height: 20;
        background: $surface;
        color: $text;
    }
    
    OptionList > .option-list--option-highlighted {
        background: $primary 20%;
    }
    
    #view-container {
        layout: vertical;
        height: 100%;
    }
    
    .metadata-panel {
        height: auto;
        min-height: 6;
        max-height: 8;
    }

    .metadata-layout {
        height: auto;
        width: 100%;
    }

    #metadata-summary {
        width: 2fr;
        padding: 0 1;
    }

    #metadata-metrics {
        width: 1.5fr;
        padding: 0 1;
        color: $text;
    }

    #metadata-reward {
        width: 1fr;
        padding: 0 1;
        text-align: left;
    }
    
    .view-columns {
        height: 1fr;
        layout: horizontal;
    }
    
    .rollouts-panel {
        width: 34;
        height: 100%;
        layout: vertical;
    }

    #rollout-list {
        height: 1fr;
        max-height: 100%;
        background: $surface;
    }

    .history-panel {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }

    .logs-panel {
        width: 1fr;
        height: 100%;
        layout: vertical;
        display: none;
    }

    #logs-scroll {
        layout: vertical;
        height: 1fr;
        background: $surface;
        padding: 0 1;
        scrollbar-size-vertical: 2;
        scrollbar-color: $primary 40%;
        scrollbar-color-hover: $primary 70%;
        scrollbar-color-active: $accent;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        scrollbar-corner-color: $panel;
    }

    #logs-scroll:focus {
        background-tint: $foreground 4%;
    }

    .column-header {
        height: auto;
        margin-bottom: 0;
        text-align: left;
        text-style: bold;
    }
    
    #completion-scroll {
        layout: vertical;
        height: 1fr;
        background: $surface;
        padding: 0 1;
        scrollbar-size-vertical: 2;
        scrollbar-color: $primary 40%;
        scrollbar-color-hover: $primary 70%;
        scrollbar-color-active: $accent;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        scrollbar-corner-color: $panel;
    }

    .history-section {
        margin: 0 0 1 0;
        background: $surface;
        border: round $secondary;
    }

    .history-section:focus-within {
        background-tint: $foreground 4%;
    }

    .history-section.just-expanded > CollapsibleTitle {
        background: $primary 18%;
        color: $text;
    }

    .history-section.expand-settle > CollapsibleTitle {
        background: $primary 10%;
        color: $text;
    }

    .history-section > CollapsibleTitle {
        text-style: bold;
        padding: 0 1;
    }

    .history-section > CollapsibleTitle:hover {
        background: $primary 12%;
        color: $text;
    }

    .history-section > CollapsibleTitle:focus {
        background: $primary 28%;
        color: $text;
    }

    .assistant-section {
        background: $success 6%;
        border: round $success;
    }

    .assistant-section > CollapsibleTitle {
        color: $success;
    }

    .tool-section {
        background: $warning 6%;
        border: round $warning;
    }

    .tool-section > CollapsibleTitle {
        color: $warning;
    }

    .prompt-section {
        background: $secondary 4%;
        border: round $secondary;
    }

    .prompt-section > CollapsibleTitle {
        color: $secondary;
    }

    .prompt-section .section-body {
        color: $text-muted;
    }

    .tool-call-section {
        background: $accent 8%;
        border: round $accent;
    }

    .tool-call-section > CollapsibleTitle {
        color: $accent;
    }

    .nested-section {
        margin: 0 0 0 1;
    }

    .section-body {
        padding: 0 1 0 1;
        color: $text;
    }

    .details-panel {
        width: 38;
        height: 1fr;
    }

    .details-scroll:focus {
        background-tint: $foreground 4%;
    }

    #details-tabs {
        height: 1fr;
    }

    #details-tabs > ContentTabs {
        background: $panel;
        margin: 0 0 1 0;
    }

    #details-tabs Tab {
        background: $surface;
        color: $text-muted;
        min-width: 8;
    }

    #details-tabs Tab.-active {
        color: $text;
    }

    #details-tabs ContentSwitcher {
        height: 1fr;
    }

    #details-tabs TabPane {
        height: 1fr;
        padding: 0;
    }

    .surface-scroll {
        height: 1fr;
        background: $surface;
        padding: 0 1;
        scrollbar-color: $secondary;
        scrollbar-background: $panel;
        scrollbar-corner-color: $panel;
    }

    #run-browser-details-scroll {
        padding: 0 1 0 2;
        scrollbar-size-vertical: 2;
        scrollbar-gutter: stable;
    }

    #run-browser-details {
        margin-right: 8;
    }

    #compare-scroll {
        padding: 0 1 0 2;
        scrollbar-size-vertical: 2;
        scrollbar-gutter: stable;
    }

    #compare-content {
        margin-right: 8;
    }

    .browser-columns {
        height: 1fr;
        layout: horizontal;
    }

    .browser-tree-panel {
        width: 56;
        height: 1fr;
        layout: vertical;
    }

    #run-browser-tree {
        height: 1fr;
        background: $surface;
        color: $text;
        overflow-x: hidden;
    }

    #run-browser-tree:focus {
        background-tint: $foreground 4%;
    }

    .browser-details-panel {
        height: 1fr;
        width: 1fr;
    }

    #run-browser-details-scroll:focus {
        background-tint: $foreground 4%;
    }

    .compare-panel {
        height: 1fr;
    }

    Footer {
        background: $panel;
    }
    
    .modal-header {
        height: auto;
    }
    
    .modal-columns {
        height: 1fr;
        layout: horizontal;
    }
    
    .modal-panel {
        width: 50%;
        height: 100%;
        layout: vertical;
    }

    .compact-copy-body {
        height: 1fr;
        layout: vertical;
    }

    .search-input {
        background: $surface;
        color: $text;
    }

    .copy-targets {
        height: 1fr;
        background: $surface;
        color: $text;
    }

    .copy-textarea {
        height: 1fr;
        background: $surface;
        color: $text;
    }

    """

    def __init__(self, index: Dict[str, Dict[str, List[RunInfo]]]):
        super().__init__()
        self.index = index

    def on_mount(self) -> None:
        # Register both custom themes
        pass

    async def action_quit(self) -> None:
        """Quit the application."""
        pass

    def action_toggle_dark(self) -> None:
        """Toggle between dark and light themes."""
        pass


class SearchScreen(ModalScreen[Optional[SearchResult]]):
    """Modal screen for searching prompt/completion text."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(
        self,
        prompt_lines: List[Tuple[int, int, str]],
        completion_lines: List[Tuple[int, int, str]],
    ):
        super().__init__()
        self._tagged_lines: Dict[str, List[Tuple[int, int, str]]] = {
            "prompt": prompt_lines,
            "completion": completion_lines,
        }
        self._hits: Dict[str, List[SearchHit]] = {
            "prompt": [],
            "completion": [],
        }
        self._cursors: Dict[str, Optional[int]] = {
            "prompt": None,
            "completion": None,
        }
        self._active_column: Optional[str] = None

    def compose(self) -> ComposeResult:
        pass

    def on_mount(self) -> None:
        pass

    def on_input_changed(self, event: Input.Changed) -> None:
        pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        pass

    @on(OptionList.OptionHighlighted, "#prompt-results")
    def on_prompt_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        pass

    @on(OptionList.OptionHighlighted, "#completion-results")
    def on_completion_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        pass

    @on(OptionList.OptionSelected, "#prompt-results")
    def on_prompt_selected(self, event: OptionList.OptionSelected) -> None:
        pass

    @on(OptionList.OptionSelected, "#completion-results")
    def on_completion_selected(self, event: OptionList.OptionSelected) -> None:
        pass

    def on_key(self, event) -> None:
        pass

    def action_close(self) -> None:
        pass

    def action_select(self) -> None:
        pass

    def _set_active_hit(
        self, column: str, option_id: Optional[str], *, select: bool = False
    ) -> None:
        pass

    def _update_results(self, pattern: str) -> None:
        pass

    def _sync_highlights(self) -> None:
        pass

    def _switch_column(self, target: str) -> None:
        pass

    def _move_selection(self, delta: int) -> None:
        pass

    def _current_selection(self) -> Optional[SearchHit]:
        pass


class RolloutCopyScreen(ModalScreen[None]):
    """Modal screen for copying rollout viewer sections."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "close", "Back (esc/b)"),
        Binding("b,backspace", "close", show=False),
        Binding("c", "copy", "Copy"),
    ]

    async def action_quit(self) -> None:
        pass

    def __init__(
        self,
        items: List[RolloutCopyItem],
        *,
        start_key: Optional[str] = None,
        title: str = "Copy Rollout",
    ):
        super().__init__()
        self._items = items
        self._title = title
        self._current_idx = 0
        if start_key:
            for i, item in enumerate(items):
                if item.key == start_key:
                    self._current_idx = i
                    break
        self._last_copied_selection = ""

    def compose(self) -> ComposeResult:
        pass

    def on_mount(self) -> None:
        pass

    @on(OptionList.OptionHighlighted, "#rollout-copy-targets")
    def _on_target_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        pass

    @on(OptionList.OptionSelected, "#rollout-copy-targets")
    def _on_target_selected(self, event: OptionList.OptionSelected) -> None:
        """Click on a target: update preview and return focus to TextArea."""
        pass

    @on(TextArea.SelectionChanged)
    def _on_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        pass

    def on_key(self, event: events.Key) -> None:
        # Only intercept arrow keys when the OptionList has focus;
        # let all keys pass through to the TextArea normally.
        pass

    def _move_section(self, delta: int) -> None:
        pass

    def action_close(self) -> None:
        pass

    def action_copy(self) -> None:
        pass

    def _sync_preview(self) -> None:
        pass


class CompactCopyScreen(ModalScreen[None]):
    """Compact copy screen with section tabs above a preview area."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "close", "Back (esc/b)"),
        Binding("b,backspace", "close", show=False),
        Binding("tab", "next_section", "Next section"),
        Binding("shift+tab", "prev_section", "Prev section"),
        Binding("c", "copy", "Copy"),
    ]

    async def action_quit(self) -> None:
        pass

    def __init__(
        self,
        items: List[RolloutCopyItem],
        *,
        start_key: Optional[str] = None,
        title: str = "Copy",
    ):
        super().__init__()
        self._items = items
        self._title = title
        self._current_idx = 0
        if start_key:
            for i, item in enumerate(items):
                if item.key == start_key:
                    self._current_idx = i
                    break
        self._last_copied_selection = ""

    def compose(self) -> ComposeResult:
        pass

    def on_mount(self) -> None:
        pass

    @on(TextArea.SelectionChanged)
    def _on_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        pass

    def action_close(self) -> None:
        pass

    def on_key(self, event: events.Key) -> None:
        pass

    def action_prev_section(self) -> None:
        pass

    def action_next_section(self) -> None:
        pass

    def action_copy(self) -> None:
        pass

    def _sync(self) -> None:
        pass


class CopyScreen(ModalScreen[None]):
    """Modal screen for selecting and copying prompt/completion text."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "close", "Back (esc/b)"),
        Binding("b,backspace", "close", show=False),
        Binding("tab", "cycle_column", "Next column"),
        Binding("shift+tab", "cycle_column", "Prev column"),
        Binding("c", "copy", "Copy"),
    ]

    async def action_quit(self) -> None:
        pass

    def __init__(
        self,
        prompt_text: str,
        completion_text: str,
        start_column: str,
        *,
        prompt_label: str = "Prompt",
        completion_label: str = "Completion",
        title: str = "Copy Mode",
    ):
        super().__init__()
        self._prompt_text = prompt_text
        self._completion_text = completion_text
        self._prompt_label = prompt_label
        self._completion_label = completion_label
        self._title = title
        self._active_column = (
            start_column if start_column in ("prompt", "completion") else "completion"
        )
        self._last_copied_selection = ""

    def compose(self) -> ComposeResult:
        pass

    def on_mount(self) -> None:
        pass

    @on(TextArea.SelectionChanged)
    def _on_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        pass

    def action_close(self) -> None:
        pass

    def on_key(self, event) -> None:
        pass

    def action_cycle_column(self) -> None:
        pass

    def action_copy(self) -> None:
        pass

    def _active_text_area(self) -> TextArea:
        pass

    def _refresh_ui(self, *, focus_text_area: bool = False) -> None:
        pass


def main() -> None:
    env_dir = os.environ.get("VF_ENV_DIR", "./environments")
    outputs_dir = os.environ.get("VF_OUTPUTS_DIR", "./outputs")
    VerifiersTUI(discover_results(env_dir, outputs_dir)).run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Memora Versioning with semantic adaptation and full-operation coverage."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

_TOOL_DIR = Path(__file__).resolve().parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

from memora_adapter import (  # noqa: E402
    MemoraVersioningAdapter,
    SemanticMemoraVersioningAdapter,
    build_full_capability_records,
    write_jsonl,
)
from statebudgetmem.schemas import MemoryRecord  # noqa: E402
from statebudgetmem.versioning import UpdateOperation, VersioningEngine  # noqa: E402


_OPERATION_ZH = {
    "ADD": "新增",
    "MERGE": "合并补充",
    "SUPERSEDE": "永久替代",
    "TEMP_INVALIDATE": "临时失效",
    "RESTORE": "恢复",
    "DELETE": "删除",
    "NOOP": "无操作",
}
_ALL_OPERATIONS = [item.value for item in UpdateOperation]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行 Memora 真实数据，并单独展示 Versioning 全操作能力。"
    )
    parser.add_argument("--memora-dir", required=True)
    parser.add_argument("--period", default="weekly", choices=("weekly", "monthly", "quarterly"))
    parser.add_argument("--persona", default="software_engineer")
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--mode", choices=("label", "semantic"), default="semantic")
    parser.add_argument(
        "--report-mode",
        choices=("changes", "all", "mismatches"),
        default="changes",
        help="报告主表显示变化操作、全部操作或声明/实际不一致操作。",
    )
    parser.add_argument("--report-rows", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    real_dir = output_dir / "memora_real"
    coverage_dir = output_dir / "controlled_full_coverage"
    real_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)

    adapter = (
        SemanticMemoraVersioningAdapter()
        if args.mode == "semantic"
        else MemoraVersioningAdapter()
    )
    conversion = adapter.convert_persona(
        args.memora_dir,
        period=args.period,
        persona=args.persona,
        limit_sessions=args.limit_sessions,
    )
    real_result = _run_records(conversion.records, real_dir, source_kind="memora_real")
    write_jsonl(real_dir / "conversion_issues.jsonl", (item.model_dump() for item in conversion.issues))

    coverage_records = build_full_capability_records()
    coverage_result = _run_records(
        coverage_records,
        coverage_dir,
        source_kind="controlled_full_coverage",
    )

    covered = set(coverage_result["summary"]["operation_counts"])
    missing = [operation for operation in _ALL_OPERATIONS if operation not in covered]
    if missing:
        raise RuntimeError(f"controlled coverage suite failed to cover operations: {missing}")

    summary = {
        "mode": args.mode,
        "period": args.period,
        "persona": args.persona,
        "scanned_sessions": conversion.scanned_sessions,
        "converted_records": len(conversion.records),
        "conversion_issues": len(conversion.issues),
        "memora": real_result["summary"],
        "controlled_coverage": coverage_result["summary"],
        "all_operations_covered": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.html").write_text(
        _render_report(
            summary,
            real_result["decisions"],
            coverage_result["decisions"],
            report_mode=args.report_mode,
            report_rows=max(1, args.report_rows),
        ),
        encoding="utf-8",
    )

    print("Memora + Versioning 全能力展示运行完成")
    print(f"  模式                  : {args.mode}")
    print(f"  Memora session        : {conversion.scanned_sessions}")
    print(f"  Memora 转换记录       : {len(conversion.records)}")
    print(f"  Memora 操作分布       : {real_result['summary']['operation_counts']}")
    print(f"  受控全操作覆盖        : {coverage_result['summary']['operation_counts']}")
    print(f"  全部 7 类操作覆盖     : 是")
    print(f"  Memora 运行错误       : {real_result['summary']['runtime_errors']}")
    print(f"  受控套件运行错误      : {coverage_result['summary']['runtime_errors']}")
    print(f"  中文报告              : {output_dir / 'report.html'}")
    return 0


def _run_records(
    records: Iterable[MemoryRecord],
    output_dir: Path,
    *,
    source_kind: str,
) -> dict[str, Any]:
    materialized = list(records)
    write_jsonl(output_dir / "processed_records.jsonl", (item.model_dump(mode="json") for item in materialized))

    engine = VersioningEngine()
    decisions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record in materialized:
        try:
            result = engine.ingest(record).results[0]
            decisions.append(
                {
                    "source_kind": source_kind,
                    "memory_id": record.memory_id,
                    "date": record.event_time.isoformat(),
                    "attribute": record.attribute,
                    "value": record.value,
                    "source_operation": record.metadata.get("memora_operation"),
                    "declared_intent": record.metadata.get("versioning_intent"),
                    "actual_operation": result.decision.operation.value,
                    "targets": result.decision.target_memory_ids,
                    "confidence": result.decision.confidence,
                    "requires_review": result.decision.requires_review,
                    "reason": result.decision.reason,
                    "adapter_reason": record.metadata.get("adapter_inference_reason"),
                    "created_node_ids": result.created_node_ids,
                    "updated_node_ids": result.updated_node_ids,
                    "created_edges": [item.model_dump(mode="json") for item in result.created_edges],
                    "skipped": result.skipped,
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "memory_id": record.memory_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    write_jsonl(output_dir / "decisions.jsonl", decisions)
    write_jsonl(output_dir / "runtime_errors.jsonl", errors)
    _write_json(output_dir / "version_graph.json", engine.snapshot())

    counts = Counter(item["actual_operation"] for item in decisions)
    summary = {
        "record_count": len(materialized),
        "decision_count": len(decisions),
        "runtime_errors": len(errors),
        "graph_nodes": len(engine.graph.nodes),
        "graph_edges": len(engine.graph.edges),
        "operation_counts": {key: counts.get(key, 0) for key in _ALL_OPERATIONS if counts.get(key, 0)},
        "requires_review_count": sum(1 for item in decisions if item["requires_review"]),
        "graph_valid": engine.validate().is_valid,
    }
    _write_json(output_dir / "summary.json", summary)
    return {"summary": summary, "decisions": decisions}


def _render_report(
    summary: dict[str, Any],
    real_decisions: list[dict[str, Any]],
    coverage_decisions: list[dict[str, Any]],
    *,
    report_mode: str,
    report_rows: int,
) -> str:
    selected_real = _select_decisions(real_decisions, mode=report_mode, limit=report_rows)
    real_counts = summary["memora"]["operation_counts"]
    coverage_counts = summary["controlled_coverage"]["operation_counts"]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StateBudgetMem Versioning 全能力展示</title>
<style>
body {{ font-family: system-ui, -apple-system, 'Microsoft YaHei', sans-serif; max-width: 1180px; margin: 32px auto; padding: 0 20px; line-height: 1.6; color: #1f2937; }}
h1,h2 {{ line-height: 1.25; }}
.notice {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 12px; padding: 14px 16px; }}
.good {{ background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 12px; padding: 14px 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
.card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; }}
.value {{ font-size: 1.45rem; font-weight: 700; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th,td {{ border-bottom: 1px solid #e5e7eb; padding: 9px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f9fafb; position: sticky; top: 0; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2ff; }}
.small {{ color:#6b7280; font-size:13px; }}
</style>
</head>
<body>
<h1>StateBudgetMem Versioning 全能力展示</h1>
<p class="notice"><strong>证据分层：</strong>“Memora 真实数据”用于展示真实规模接入和语义适配；“受控全操作套件”用于保证 ADD、MERGE、SUPERSEDE、TEMP_INVALIDATE、RESTORE、DELETE、NOOP 七类能力全部出现。两者不会混在一起冒充同一种数据结果。</p>

<h2>一、Memora 真实数据</h2>
<div class="grid">
{_metric_card('扫描 session', summary['scanned_sessions'])}
{_metric_card('转换记录', summary['converted_records'])}
{_metric_card('图节点', summary['memora']['graph_nodes'])}
{_metric_card('图边', summary['memora']['graph_edges'])}
{_metric_card('运行错误', summary['memora']['runtime_errors'])}
{_metric_card('需复核', summary['memora']['requires_review_count'])}
</div>
<p><strong>真实数据操作分布：</strong>{html.escape(_counts_text(real_counts))}</p>
<p class="small">语义模式只在有可观察证据时推断合并、临时失效、恢复或无操作；如果 Memora 没有相应事件，这些操作可能仍然为 0。</p>
{_decision_table(selected_real, limit=report_rows)}

<h2>二、受控全操作能力套件</h2>
<p class="good"><strong>七类操作已全部覆盖：</strong>{html.escape(_counts_text(coverage_counts))}</p>
{_decision_table(coverage_decisions, limit=20)}

<h2>三、正确解读</h2>
<ul>
<li>Memora 部分证明：真实数据能够稳定进入 Versioning，并形成版本图。</li>
<li>受控套件证明：Versioning 的七类更新语义都能被实际触发和执行。</li>
<li>这份报告不等同于操作分类准确率；准确率仍需要人工标注或金标准。</li>
</ul>
</body>
</html>"""


def _metric_card(label: str, value: Any) -> str:
    return f'<div class="card"><div class="small">{html.escape(label)}</div><div class="value">{html.escape(str(value))}</div></div>'


def _counts_text(counts: dict[str, int]) -> str:
    return "；".join(f"{_OPERATION_ZH.get(key, key)} {value}" for key, value in counts.items()) or "无"


def _select_decisions(
    decisions: list[dict[str, Any]],
    *,
    mode: str,
    limit: int,
) -> list[dict[str, Any]]:
    if mode == "changes":
        selected = [item for item in decisions if item["actual_operation"] != "ADD"]
    elif mode == "mismatches":
        selected = [
            item for item in decisions
            if item["actual_operation"] != item.get("declared_intent")
        ]
    else:
        selected = decisions
    return selected[:limit]


def _decision_table(rows: list[dict[str, Any]], *, limit: int) -> str:
    body = []
    for row in rows[:limit]:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['date']))}</td>"
            f"<td><span class='badge'>{html.escape(_OPERATION_ZH.get(row['actual_operation'], row['actual_operation']))}</span></td>"
            f"<td>{html.escape(str(row['attribute']))}</td>"
            f"<td>{html.escape(str(row['value'])[:180])}</td>"
            f"<td>{html.escape(', '.join(row['targets']) or '-')}</td>"
            f"<td>{html.escape(str(row.get('adapter_reason') or row.get('reason') or ''))}</td>"
            "</tr>"
        )
    if not body:
        body.append("<tr><td colspan='6'>无记录</td></tr>")
    return (
        "<div style='overflow:auto;max-height:560px'><table>"
        "<thead><tr><th>日期</th><th>操作</th><th>属性</th><th>值</th><th>目标</th><th>原因</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

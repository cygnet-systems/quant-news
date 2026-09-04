"""Multi-sheet XLSX export of everything currently on screen.

The old export wrote a Parquet of the FIRST selected symbol's raw prices,
not openable by most users and silently ignoring the rest of the screen.
This builds one workbook that mirrors the display: per-symbol price sheets
carry OHLCV plus ONLY the indicators the user has toggled on (MACD/RSI are
always included because their panes always render), and Predictions /
AI Analysis / Recommendations sheets appear only when those stores hold
data: the workbook grows with whatever the session has produced.
"""

from typing import Optional
import io
import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

# Chart-toggle value -> the indicator columns it displays
_INDICATOR_COLUMNS: dict[str, list[str]] = {
    "sma_20": ["SMA_20"],
    "sma_50": ["SMA_50"],
    "sma_200": ["SMA_200"],
    "bollinger": ["BB_Upper", "BB_Mid", "BB_Lower"],
}

# Panes that are always on screen regardless of toggles
_ALWAYS_COLUMNS = ["RSI", "MACD", "MACD_Signal", "MACD_Hist"]

_BASE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _sheet_name(name: str) -> str:
    """Excel sheet names: max 31 chars, no []:*?/\\ characters."""
    for ch in "[]:*?/\\":
        name = name.replace(ch, "-")
    return name[:31]


def _autosize(writer: pd.ExcelWriter, sheet: str, df: pd.DataFrame) -> None:
    try:
        ws = writer.sheets[sheet]
        for i, col in enumerate(df.columns, start=1):
            width = max(len(str(col)), *(len(str(v)) for v in df[col].head(80)))
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = (
                min(max(width + 2, 10), 60)
            )
    except Exception:  # cosmetic only: never fail an export over widths
        pass


def build_xlsx(
    symbols: list[str],
    stock_data: dict,
    selected_indicators: list[str] | None = None,
    model_signals: dict | None = None,
    ai_analysis: dict | None = None,
    recommendations: dict | None = None,
) -> bytes:
    """Assemble the workbook. Sheets appear only when their data exists."""
    selected_indicators = selected_indicators or []
    buf = io.BytesIO()

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # --- Summary: one row per symbol, mirroring the metric cards ---
        summary_rows = []
        for sym in symbols:
            entry = (stock_data or {}).get(sym) or {}
            metrics = entry.get("metrics") or {}
            signals = entry.get("signals") or {}
            row = {
                "Symbol": sym,
                "Close": metrics.get("end_price"),
                "Period Return %": metrics.get("total_return"),
                "Volatility % (ann.)": metrics.get("volatility"),
                "Max Drawdown %": metrics.get("max_drawdown"),
                "Period": f"{metrics.get('start_date', '?')} → {metrics.get('end_date', '?')}",
            }
            for key, val in signals.items():
                row[f"Signal: {key}"] = str(val)
            ana = ((ai_analysis or {}).get("by_symbol") or {}).get(sym) or {}
            if ana:
                row["AI Recommendation"] = ana.get("recommendation")
                row["AI Confidence"] = ana.get("confidence")
            rec = ((recommendations or {}).get("by_symbol") or {}).get(sym) or {}
            if rec:
                row["Synthesis Action"] = rec.get("action")
                row["Synthesis Conviction"] = rec.get("conviction")
            summary_rows.append(row)
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            df.to_excel(writer, sheet_name="Summary", index=False)
            _autosize(writer, "Summary", df)

        # --- Per-symbol prices + displayed indicators only ---
        indicator_cols: list[str] = []
        for toggle in selected_indicators:
            indicator_cols += _INDICATOR_COLUMNS.get(toggle, [])
        for sym in symbols:
            entry = (stock_data or {}).get(sym) or {}
            prices_json = entry.get("prices")
            if not prices_json:
                continue
            try:
                df = pd.read_json(io.StringIO(prices_json))
            except Exception as e:
                logger.warning(f"Export: could not parse prices for {sym}: {e}")
                continue
            df.index.name = "Date"
            cols = [c for c in _BASE_COLUMNS + indicator_cols + _ALWAYS_COLUMNS
                    if c in df.columns]
            out = df[cols].round(4)
            sheet = _sheet_name(f"{sym} Prices")
            out.to_excel(writer, sheet_name=sheet)
            _autosize(writer, sheet, out.reset_index())

        # --- Predictions (present only after a Predict run) ---
        pred_rows = []
        for sym, models in (model_signals or {}).items():
            if not isinstance(models, dict):
                continue
            for model_name, r in models.items():
                if not isinstance(r, dict):
                    continue
                pred_rows.append({
                    "Symbol": sym,
                    "Model": model_name,
                    "Decision": r.get("decision"),
                    "Confidence": r.get("confidence"),
                    "Up Probability": r.get("up_probability"),
                    "Error": r.get("error"),
                })
        if pred_rows:
            df = pd.DataFrame(pred_rows).sort_values(["Symbol", "Model"])
            df.to_excel(writer, sheet_name="Predictions", index=False)
            _autosize(writer, "Predictions", df)

        # --- AI Analysis (per symbol + overall) ---
        ai_rows = []
        for sym, ana in ((ai_analysis or {}).get("by_symbol") or {}).items():
            if not isinstance(ana, dict):
                continue
            ai_rows.append({
                "Scope": sym,
                "Recommendation": ana.get("recommendation"),
                "Confidence": ana.get("confidence"),
                "Key Developments": ana.get("key_developments"),
                "Read": ana.get("developments_read"),
                "Sentiment": ana.get("market_sentiment"),
                "Risk Factors": ana.get("risk_factors"),
                "Risk Read": ana.get("risks_read"),
                "Watch Items": "; ".join(ana.get("watch_items") or [])
                               if isinstance(ana.get("watch_items"), list) else None,
            })
        overall = (ai_analysis or {}).get("overall")
        if isinstance(overall, dict) and overall:
            ai_rows.append({
                "Scope": "OVERALL",
                "Recommendation": overall.get("recommendation"),
                "Confidence": overall.get("confidence"),
                "Key Developments": overall.get("key_developments"),
                "Read": overall.get("developments_read"),
                "Sentiment": overall.get("market_sentiment"),
                "Risk Factors": overall.get("risk_factors"),
                "Risk Read": overall.get("risks_read"),
                "Watch Items": "; ".join(overall.get("watch_items") or [])
                               if isinstance(overall.get("watch_items"), list) else None,
            })
        if ai_rows:
            df = pd.DataFrame(ai_rows)
            df.to_excel(writer, sheet_name="AI Analysis", index=False)
            _autosize(writer, "AI Analysis", df)

        # --- recommendation synthesis ---
        rec_rows = []
        for sym, rec in ((recommendations or {}).get("by_symbol") or {}).items():
            if not isinstance(rec, dict):
                continue
            rec_rows.append({
                "Symbol": sym,
                "Action": rec.get("action"),
                "Conviction": rec.get("conviction"),
                "Reasoning": rec.get("reasoning"),
                "Key Level": rec.get("key_level"),
                "Flips On": rec.get("change_trigger"),
                "Model Notes": rec.get("model_notes"),
            })
        rec_overall = (recommendations or {}).get("overall") or {}
        if rec_rows or rec_overall:
            df = pd.DataFrame(rec_rows) if rec_rows else pd.DataFrame()
            if rec_overall:
                meta = pd.DataFrame([{
                    "Symbol": "PORTFOLIO",
                    "Action": rec_overall.get("portfolio_action"),
                    "Conviction": None,
                    "Reasoning": rec_overall.get("summary"),
                    "Key Level": None,
                    "Flips On": None,
                    "Model Notes": rec_overall.get("risk_assessment"),
                }])
                df = pd.concat([meta, df], ignore_index=True) if not df.empty else meta
            df.to_excel(writer, sheet_name="Recommendations", index=False)
            _autosize(writer, "Recommendations", df)

        # --- Export metadata ---
        meta = pd.DataFrame([
            {"Field": "Exported at", "Value": datetime.now().isoformat(timespec="seconds")},
            {"Field": "Symbols", "Value": ", ".join(symbols)},
            {"Field": "Indicators shown", "Value": ", ".join(selected_indicators) or "none"},
            {"Field": "Sheets grow with on-screen data",
             "Value": "Predictions/AI/Recommendations appear once generated"},
        ])
        meta.to_excel(writer, sheet_name="About", index=False)
        _autosize(writer, "About", meta)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Model-input provenance workbook
# ---------------------------------------------------------------------------

def _news_rows(symbol: str, articles: list) -> list[dict]:
    """Rows for the News sheet from already-computed article fields.

    Deliberately NO LLM involvement: sentiment/relevance/topics were computed
    when the articles were ingested; a download must never incur model cost.
    """
    rows = []
    for a in articles:
        get = (lambda k, d=None: getattr(a, k, d)) if hasattr(a, "title") \
            else (lambda k, d=None: a.get(k, d))
        topics = get("topics")
        if isinstance(topics, (list, tuple)):
            topics = ", ".join(str(t) for t in topics)
        rows.append({
            "Symbol": symbol,
            "Published": str(get("published_at") or "")[:16],
            "Source": get("source"),
            "Sentiment": get("sentiment"),
            "Sent. Score": get("sentiment_score"),
            "Relevance": get("ticker_relevance_score"),
            "Title": get("title"),
            "URL": get("url"),
            "Topics": topics,
            "Summary": (get("summary") or "")[:400],
        })
    return rows


def _linkify_url_column(writer: pd.ExcelWriter, sheet: str, df: pd.DataFrame) -> None:
    """Turn the URL column into real Excel hyperlinks (openpyxl post-pass)."""
    try:
        if "URL" not in df.columns:
            return
        ws = writer.sheets[sheet]
        col = list(df.columns).index("URL") + 1
        for r in range(2, len(df) + 2):
            cell = ws.cell(row=r, column=col)
            if cell.value:
                cell.hyperlink = str(cell.value)
                cell.style = "Hyperlink"
    except Exception:  # cosmetic only
        pass


def build_model_inputs_xlsx(
    symbols: list[str],
    as_of: str,
    news_lookback_days: Optional[int] = None,
    max_articles: int = 0,
) -> bytes:
    """Workbook of the point-in-time inputs behind a report/prediction/rec.

    One sheet per model input class (symbols stacked with a Symbol column):
    Kronos OHLCV, XGB-LGBM feature vector, the news window (with links and
    the precomputed sentiment fields), the research agent's text blocks, and
    the model signals recorded for that date. Everything is reconstructed
    with the SAME lookahead-safe builders the models use. Sliced to as_of,
    so users can audit the inputs and draw their own conclusions.
    """
    from services.news_window import RunParameterMissing
    if news_lookback_days is None:
        raise RunParameterMissing("model-inputs export needs the run's news "
                                  "window; none was supplied")
    from datetime import datetime as _dt

    from models.feature_builder import FEATURE_VERSION, LiveFeatureBuilder
    from models.sector_map import get_sector_info
    from models.single_agent import (
        _as_of_slice, _fundamentals_block, _price_action_block,
        _technicals_block,
    )
    from services.news_window import fetch_point_in_time_news
    from services.stock_data import fetch_stock_data, get_company_profile

    as_of = str(as_of)[:10]
    # as_of is the data cutoff; the target is the next session's close.
    try:
        from utils.trading_calendar import get_next_trading_day
        _target_str = str(get_next_trading_day(as_of))
    except Exception:
        _target_str = "N/A"
    buf = io.BytesIO()

    spy_full = fetch_stock_data("SPY", period="2y")
    spy_df = _as_of_slice(spy_full, as_of)

    ohlcv_rows, feature_rows, news_all, block_rows = [], [], [], []
    for sym in symbols:
        try:
            tdf = _as_of_slice(fetch_stock_data(sym, period="1y"), as_of)
        except Exception:
            tdf = None
        if tdf is None or not len(tdf):
            block_rows.append({"Symbol": sym, "Block": "ERROR",
                               "Content": f"No OHLCV available through {as_of}"})
            continue

        # --- Kronos / price input: the daily OHLCV window ---
        for idx, row in tdf.tail(120).iterrows():
            ohlcv_rows.append({
                "Symbol": sym, "Date": str(idx)[:10],
                "Open": round(float(row["Open"]), 4),
                "High": round(float(row["High"]), 4),
                "Low": round(float(row["Low"]), 4),
                "Close": round(float(row["Close"]), 4),
                "Volume": int(row["Volume"]) if pd.notna(row.get("Volume")) else None,
            })

        # --- news window (links + precomputed sentiment features) ---
        try:
            articles = fetch_point_in_time_news(sym, as_of,
                                                lookback_days=news_lookback_days,
                                                max_articles=max_articles)
        except Exception:
            articles = []
        news_all.extend(_news_rows(sym, articles))

        # --- XGB/LGBM feature vector ---
        sinfo = get_sector_info(sym)
        try:
            sector_df = _as_of_slice(fetch_stock_data(sinfo["etf"], period="2y"), as_of) \
                if sinfo["etf"] != "SPY" else spy_df
        except Exception:
            sector_df = spy_df
        try:
            day_news = [a for a in articles
                        if str(getattr(a, "published_at", "") or "")[:10] == as_of]
            feats = LiveFeatureBuilder().build_features(
                tdf.copy(), spy_df.copy(), sector_df.copy(),
                av_news=[{"sentiment": getattr(a, "sentiment", None),
                          "sentiment_score": getattr(a, "sentiment_score", None),
                          "topics": getattr(a, "topics", None),
                          "ticker_relevance_score": getattr(a, "ticker_relevance_score", None),
                          "title": getattr(a, "title", "")} for a in day_news],
            )
            feature_rows.append({"Symbol": sym, **{k: v for k, v in feats.items()}})
        except Exception as e:
            feature_rows.append({"Symbol": sym, "error": str(e)[:120]})

        # --- research agent text blocks ---
        blocks = [("Business Profile", get_company_profile(sym) or "n/a"),
                  ("SPY Context", _technicals_block("SPY", spy_df))]
        if sinfo["etf"] != "SPY":
            blocks.append((f"Sector Context ({sinfo['etf']}, "
                           f"{sinfo['level']}-level)",
                           _technicals_block(sinfo["etf"], sector_df)))
        else:
            blocks.append(("Sector Context", "sector metadata unavailable"))
        blocks.append((f"{sym} Technicals", _technicals_block(sym, tdf)))
        blocks.append((f"{sym} Price Action", _price_action_block(tdf)))
        blocks.append((f"{sym} Fundamentals", _fundamentals_block(sym, as_of)))
        try:
            from models.trading_agents_model import TradingAgentsModel
            # (blocks, investigation, anomalies, screened): iterating the
            # tuple itself handed the whole list to .split() and every export
            # silently fell into the except below with "Precomputed blocks:
            # unavailable".
            extra_blocks, _, _, _ = TradingAgentsModel()._build_extra_context(
                sym, tdf, as_of)
            for extra in extra_blocks:
                title = extra.split("\n", 1)[0][:60]
                blocks.append((f"Precomputed: {title}", extra))
        except Exception as e:
            blocks.append(("Precomputed blocks", f"unavailable: {e}"))
        for name, content in blocks:
            block_rows.append({"Symbol": sym, "Block": name, "Content": content})

    # --- recorded model signals for that date ---
    signal_rows = []
    try:
        from services.cache_service import get_cache
        for p in get_cache().list_all_predictions(limit=2000):
            if (str(p.get("prediction_date"))[:10] == as_of
                    and p.get("symbol") in set(symbols)):
                signal_rows.append({
                    "Symbol": p.get("symbol"), "Model": p.get("model_name"),
                    "Decision": p.get("decision"),
                    "Confidence": p.get("confidence"),
                    "Up Prob": p.get("up_probability"),
                    "Target Date": p.get("target_date"),
                    "Was Correct": p.get("was_correct"),
                    "P&L $": p.get("pnl_dollars"),
                })
    except Exception:
        pass

    readme = pd.DataFrame([
        {"Field": "What is this?",
         "Value": "The point-in-time inputs behind this report/prediction, so you "
                  "can audit them and draw your own conclusions."},
        {"Field": "Data through (as-of)", "Value": as_of},
        {"Field": "Target date (close being predicted)", "Value": _target_str},
        {"Field": "Symbols", "Value": ", ".join(symbols)},
        {"Field": "Generated", "Value": _dt.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"Field": "Kronos OHLCV",
         "Value": "Daily bars through as-of (last 120 shown); Kronos consumes the "
                  "trailing window of this series."},
        {"Field": "XGB-LGBM Features",
         "Value": f"The v{FEATURE_VERSION} feature vector as the tree models see it. "
                  "Global-news topic features may read 0 in reconstruction when the "
                  "historical global feed isn't cached."},
        {"Field": "News",
         "Value": f"{news_lookback_days}d point-in-time window ending as-of. "
                  "Sentiment/relevance/topics were computed at ingestion, no LLM "
                  "ran to produce this file."},
        {"Field": "Research Context",
         "Value": "The exact text blocks assembled for the research agent "
                  "(lookahead-safe, sliced to as-of)."},
        {"Field": "Model Signals",
         "Value": "Predictions recorded in the DB for this date (empty if none)."},
        {"Field": "Caveat",
         "Value": "Reconstructed with current code against cached/vendor data; if "
                  "vendor history was revised after the report ran, values can "
                  "differ marginally from what the model saw."},
    ])

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        _autosize(writer, "README", readme)
        if ohlcv_rows:
            df = pd.DataFrame(ohlcv_rows)
            df.to_excel(writer, sheet_name="Kronos OHLCV", index=False)
            _autosize(writer, "Kronos OHLCV", df)
        if feature_rows:
            df = pd.DataFrame(feature_rows)
            df.to_excel(writer, sheet_name="XGB-LGBM Features", index=False)
            _autosize(writer, "XGB-LGBM Features", df)
        if news_all:
            df = pd.DataFrame(news_all)
            df.to_excel(writer, sheet_name="News", index=False)
            _autosize(writer, "News", df)
            _linkify_url_column(writer, "News", df)
        if block_rows:
            df = pd.DataFrame(block_rows)
            df.to_excel(writer, sheet_name="Research Context", index=False)
            _autosize(writer, "Research Context", df)
        if signal_rows:
            df = pd.DataFrame(signal_rows)
            df.to_excel(writer, sheet_name="Model Signals", index=False)
            _autosize(writer, "Model Signals", df)

    return buf.getvalue()

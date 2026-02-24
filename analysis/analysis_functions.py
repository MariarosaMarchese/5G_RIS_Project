import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def summarize_series(s: pd.Series) -> Dict[str, float]:
    s = s.dropna()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": float(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


# -----------------------------
# iperf3 UDP parsing
# -----------------------------
@dataclass
class IperfUDPResult:
    df: pd.DataFrame  # per-interval data
    summary: Dict[str, float]


def parse_iperf3_udp_text(text: str) -> IperfUDPResult:
    """
    Parses typical iperf3 UDP output (client or server).
    Extracts per-interval:
      - interval_start_s, interval_end_s
      - bitrate_Mbps
      - jitter_ms (if present)
      - lost, total (if present)
      - loss_percent (if present)
    """
    lines = text.splitlines()

    # Example UDP per-interval formats can vary. We support several patterns.
    # Common one (server side):
    # [  5]   1.00-2.00   sec  1.19 MBytes  9.97 Mbits/sec  0.043 ms  0/  852 (0%)
    interval_patterns = [
        re.compile(
            r"\[\s*\d+\]\s+"
            r"(?P<t0>\d+(?:\.\d+)?)\s*-\s*(?P<t1>\d+(?:\.\d+)?)\s*sec\s+"
            r".*?\s+(?P<bitrate>\d+(?:\.\d+)?)\s*(?P<unit>Mbits/sec|Kbits/sec|bits/sec)\s+"
            r"(?P<jitter>\d+(?:\.\d+)?)\s*ms\s+"
            r"(?P<lost>\d+)\s*/\s*(?P<total>\d+)\s*\((?P<loss_pct>\d+(?:\.\d+)?)%\)"
        ),
        # Sometimes jitter/loss missing on some lines or different spacing:
        re.compile(
            r"\[\s*\d+\]\s+"
            r"(?P<t0>\d+(?:\.\d+)?)\s*-\s*(?P<t1>\d+(?:\.\d+)?)\s*sec\s+"
            r".*?\s+(?P<bitrate>\d+(?:\.\d+)?)\s*(?P<unit>Mbits/sec|Kbits/sec|bits/sec)"
            r"(?:\s+(?P<jitter>\d+(?:\.\d+)?)\s*ms\s+(?P<lost>\d+)\s*/\s*(?P<total>\d+)\s*\((?P<loss_pct>\d+(?:\.\d+)?)%\))?"
        ),
    ]

    rows = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("iperf3:"):
            continue

        m = None
        for pat in interval_patterns:
            m = pat.search(ln)
            if m:
                break
        if not m:
            continue

        t0 = safe_float(m.group("t0"))
        t1 = safe_float(m.group("t1"))
        bitrate = safe_float(m.group("bitrate"))
        unit = m.group("unit")

        # Normalize bitrate to Mbps
        bitrate_mbps = None
        if bitrate is not None:
            if unit == "Mbits/sec":
                bitrate_mbps = bitrate
            elif unit == "Kbits/sec":
                bitrate_mbps = bitrate / 1000.0
            elif unit == "bits/sec":
                bitrate_mbps = bitrate / 1e6

        jitter_ms = safe_float(m.group("jitter")) if m.groupdict().get("jitter") else None
        lost = float(m.group("lost")) if m.groupdict().get("lost") else np.nan
        total = float(m.group("total")) if m.groupdict().get("total") else np.nan
        loss_pct = safe_float(m.group("loss_pct")) if m.groupdict().get("loss_pct") else None

        rows.append(
            {
                "interval_start_s": t0,
                "interval_end_s": t1,
                "bitrate_Mbps": bitrate_mbps,
                "jitter_ms": jitter_ms,
                "lost": lost,
                "total": total,
                "loss_percent": loss_pct,
                "raw_line": ln,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return IperfUDPResult(df=df, summary={"n": 0})

    # Try to compute a goodput estimate if loss is available (UDP receiver side)
    # goodput ≈ bitrate * (1 - loss%)
    if "loss_percent" in df.columns and df["loss_percent"].notna().any():
        df["goodput_Mbps"] = df["bitrate_Mbps"] * (1.0 - df["loss_percent"] / 100.0)
    else:
        df["goodput_Mbps"] = np.nan

    # Summary stats
    summary = {}
    summary.update({f"throughput_{k}": v for k, v in summarize_series(df["bitrate_Mbps"]).items()})
    summary.update({f"jitter_{k}": v for k, v in summarize_series(df["jitter_ms"]).items()})
    summary.update({f"losspct_{k}": v for k, v in summarize_series(df["loss_percent"]).items()})
    summary.update({f"goodput_{k}": v for k, v in summarize_series(df["goodput_Mbps"]).items()})
    return IperfUDPResult(df=df, summary=summary)


def load_iperf_file(path: str) -> IperfUDPResult:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return parse_iperf3_udp_text(txt)


def trim_steady_state_iperf(df: pd.DataFrame, skip_first_s: float = 10.0, skip_last_s: float = 0.0) -> pd.DataFrame:
    """
    Remove transient (e.g., first 10s) and optionally tail.
    This addresses your professor's "specify the intervals":
      -> you can explicitly say: "metrics computed over [10s, 110s]" etc.
    """
    if df.empty:
        return df
    t_min = df["interval_start_s"].min()
    t_max = df["interval_end_s"].max()
    start = t_min + skip_first_s
    end = t_max - skip_last_s
    return df[(df["interval_start_s"] >= start) & (df["interval_end_s"] <= end)].copy()


# -----------------------------
# OAI log parsing (gNB/UE)
# -----------------------------
@dataclass
class OAILogMetrics:
    rsrp_dbm: pd.Series
    snr_db: pd.Series
    dl_bler_pct: pd.Series
    ul_bler_pct: pd.Series


def parse_oai_metrics(text: str) -> OAILogMetrics:
    """
    Parser tailored to OAI gNB logs like:

      UE ... average RSRP -89 (16 meas)
      UE ... dlsch_rounds ... BLER 0.00003 ...
      UE ... ulsch_rounds ... BLER 0.04373 ... SNR 33.0 dB ...

    Notes:
    - BLER in your logs appears as a fraction (0..1), so we convert to percent.
    """

    # RSRP line
    # Example: "average RSRP -89 (16 meas)"
    rsrp_pat = re.compile(r"\baverage\s+RSRP\s+(?P<rsrp>-?\d+(?:\.\d+)?)\b", re.IGNORECASE)

    # DL BLER line (DLSCH)
    # Example: "dlsch_rounds ... BLER 0.00003 ..."
    dl_bler_pat = re.compile(r"\bdlsch_rounds\b.*?\bBLER\s+(?P<bler>\d+(?:\.\d+)?)\b", re.IGNORECASE)

    # UL BLER + SNR line (ULSCH)
    # Example: "ulsch_rounds ... BLER 0.04373 ... SNR 33.0 dB"
    ul_bler_pat = re.compile(r"\bulsch_rounds\b.*?\bBLER\s+(?P<bler>\d+(?:\.\d+)?)\b", re.IGNORECASE)
    snr_pat = re.compile(r"\bSNR\s+(?P<snr>-?\d+(?:\.\d+)?)\s*dB\b", re.IGNORECASE)

    rsrp_vals = []
    snr_vals = []
    dl_bler_vals = []
    ul_bler_vals = []

    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue

        m = rsrp_pat.search(ln)
        if m:
            rsrp_vals.append(float(m.group("rsrp")))

        m = dl_bler_pat.search(ln)
        if m:
            # Convert BLER fraction to percent
            dl_bler_vals.append(100.0 * float(m.group("bler")))

        m = ul_bler_pat.search(ln)
        if m:
            ul_bler_vals.append(100.0 * float(m.group("bler")))

        m = snr_pat.search(ln)
        if m:
            snr_vals.append(float(m.group("snr")))

    return OAILogMetrics(
        rsrp_dbm=pd.Series(rsrp_vals, dtype="float64"),
        snr_db=pd.Series(snr_vals, dtype="float64"),
        dl_bler_pct=pd.Series(dl_bler_vals, dtype="float64"),
        ul_bler_pct=pd.Series(ul_bler_vals, dtype="float64"),
    )


def load_oai_log(path: str) -> OAILogMetrics:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return parse_oai_metrics(txt)


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_cdf(data: pd.Series, title: str, xlabel: str, outpath: str) -> None:
    data = data.dropna().values
    if len(data) == 0:
        return
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.figure()
    plt.plot(x, y)
    plt.grid(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_cdf_overlay(
    s1: pd.Series,
    s2: pd.Series,
    label1: str,
    label2: str,
    title: str,
    xlabel: str,
    outpath: str,
    xlog: bool = False,
) -> None:
    """Overlay 2 CDFs on the same plot."""
    s1 = s1.dropna().values
    s2 = s2.dropna().values
    if len(s1) == 0 or len(s2) == 0:
        return

    x1 = np.sort(s1)
    y1 = np.arange(1, len(x1) + 1) / len(x1)

    x2 = np.sort(s2)
    y2 = np.arange(1, len(x2) + 1) / len(x2)

    plt.figure()
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.grid(True)
    if xlog:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_time_series(df: pd.DataFrame, ycol: str, title: str, ylabel: str, outpath: str) -> None:
    if df.empty or ycol not in df.columns:
        return
    plt.figure()
    x = df["interval_end_s"] if "interval_end_s" in df.columns else np.arange(len(df))
    plt.plot(x, df[ycol])
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def concat_iperf_by_distance(results: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate steady-state iperf data across tests for one distance."""
    ul_all = pd.concat(
        [r["iperf_ul_df"] for r in results if isinstance(r.get("iperf_ul_df"), pd.DataFrame)],
        ignore_index=True
    )
    dl_all = pd.concat(
        [r["iperf_dl_df"] for r in results if isinstance(r.get("iperf_dl_df"), pd.DataFrame)],
        ignore_index=True
    )
    return ul_all, dl_all

def plot_boxplot_two_groups(
    data1: pd.Series,
    data2: pd.Series,
    label1: str,
    label2: str,
    title: str,
    ylabel: str,
    outpath: str,
    ylog: bool = False,
) -> None:
    """
    Creates a boxplot comparison between two groups (e.g., 1m vs 2m).
    """

    data1 = data1.dropna()
    data2 = data2.dropna()

    if len(data1) == 0 or len(data2) == 0:
        return

    plt.figure()

    plt.boxplot(
        [data1, data2],
        labels=[label1, label2],
        showfliers=True,   # show outliers (important scientifically)
        patch_artist=True
    )

    if ylog:
        plt.yscale("log")

    plt.grid(True, axis="y")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# -----------------------------
# Higher-level experiment runner
# -----------------------------
def find_tests(base_data_dir: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of (distance_label, test_name, test_path)
    Expects:
      base_data_dir/testing/1m/test1/...
      base_data_dir/testing/2m/test1/...
    """
    testing_root = os.path.join(base_data_dir, "testing")
    out = []
    for dist in ["1m", "2m"]:
        dist_dir = os.path.join(testing_root, dist)
        if not os.path.isdir(dist_dir):
            continue
        for test_name in sorted(os.listdir(dist_dir)):
            test_path = os.path.join(dist_dir, test_name)
            if os.path.isdir(test_path):
                out.append((dist, test_name, test_path))
    return out


def analyze_one_test(test_path: str, skip_first_s: float = 10.0, skip_last_s: float = 0.0) -> Dict:
    """
    Reads DL.txt, UL.txt, gnb.txt, ue.txt (if present).
    Returns a dict with:
      - iperf_ul_df / iperf_dl_df trimmed
      - summaries
      - oai metrics summaries
    """
    res: Dict = {"test_path": test_path}

    # iperf
    ul_path = os.path.join(test_path, "UL.txt")
    dl_path = os.path.join(test_path, "DL.txt")

    if os.path.exists(ul_path):
        ul = load_iperf_file(ul_path)
        ul_df_trim = trim_steady_state_iperf(ul.df, skip_first_s=skip_first_s, skip_last_s=skip_last_s)
        res["iperf_ul_df"] = ul_df_trim
        res["iperf_ul_summary"] = {**ul.summary, "steady_start_s": float(ul_df_trim["interval_start_s"].min() if not ul_df_trim.empty else np.nan),
                                   "steady_end_s": float(ul_df_trim["interval_end_s"].max() if not ul_df_trim.empty else np.nan)}
    else:
        res["iperf_ul_df"] = pd.DataFrame()
        res["iperf_ul_summary"] = {"n": 0}

    if os.path.exists(dl_path):
        dl = load_iperf_file(dl_path)
        dl_df_trim = trim_steady_state_iperf(dl.df, skip_first_s=skip_first_s, skip_last_s=skip_last_s)
        res["iperf_dl_df"] = dl_df_trim
        res["iperf_dl_summary"] = {**dl.summary, "steady_start_s": float(dl_df_trim["interval_start_s"].min() if not dl_df_trim.empty else np.nan),
                                   "steady_end_s": float(dl_df_trim["interval_end_s"].max() if not dl_df_trim.empty else np.nan)}
    else:
        res["iperf_dl_df"] = pd.DataFrame()
        res["iperf_dl_summary"] = {"n": 0}

    # OAI logs: gNB + UE
    gnb_path = os.path.join(test_path, "gnb.txt")
    ue_path = os.path.join(test_path, "ue.txt")

    metrics_all = []
    if os.path.exists(gnb_path):
        metrics_all.append(load_oai_log(gnb_path))
    if os.path.exists(ue_path):
        metrics_all.append(load_oai_log(ue_path))

    # Combine (concat series)
    if metrics_all:
        ss_rsrp = pd.concat([m.rsrp_dbm for m in metrics_all], ignore_index=True)
        snr = pd.concat([m.snr_db for m in metrics_all], ignore_index=True)
        dl_bler = pd.concat([m.dl_bler_pct for m in metrics_all], ignore_index=True)
        ul_bler = pd.concat([m.ul_bler_pct for m in metrics_all], ignore_index=True)

        res["oai_ss_rsrp_stats"] = summarize_series(ss_rsrp)
        res["oai_snr_stats"] = summarize_series(snr)
        res["oai_dl_bler_stats"] = summarize_series(dl_bler)
        res["oai_ul_bler_stats"] = summarize_series(ul_bler)
        res["oai_series"] = {"ss_rsrp_dbm": ss_rsrp, "snr_db": snr, "dl_bler_pct": dl_bler, "ul_bler_pct": ul_bler}
    else:
        res["oai_ss_rsrp_stats"] = {"n": 0}
        res["oai_snr_stats"] = {"n": 0}
        res["oai_dl_bler_stats"] = {"n": 0}
        res["oai_ul_bler_stats"] = {"n": 0}
        res["oai_series"] = {"ss_rsrp_dbm": pd.Series(dtype=float), "snr_db": pd.Series(dtype=float),
                             "dl_bler_pct": pd.Series(dtype=float), "ul_bler_pct": pd.Series(dtype=float)}

    return res


def aggregate_distance(results: List[Dict], direction: str) -> pd.DataFrame:
    """
    Build a per-test summary table for a given direction ('ul' or 'dl')
    """
    rows = []
    for r in results:
        summ = r.get(f"iperf_{direction}_summary", {})
        row = {"test_path": r["test_path"]}
        # Keep only key summary fields you likely want in tables
        for k in ["throughput_mean", "throughput_median", "throughput_p95",
                  "jitter_mean", "jitter_median", "jitter_p95",
                  "losspct_mean", "losspct_p95",
                  "goodput_mean", "goodput_median",
                  "steady_start_s", "steady_end_s"]:
            if k in summ:
                row[k] = summ[k]
        rows.append(row)
    return pd.DataFrame(rows)


def build_final_kpi_table(results: List[Dict], label: str) -> pd.DataFrame:
    """
    Produces a thesis-friendly KPI table (one row for the distance label).
    Uses medians (robust) + p5/p95 hints via columns if you want.
    """
    # Concatenate per-interval iperf data across tests
    ul_all = pd.concat([r["iperf_ul_df"] for r in results if isinstance(r.get("iperf_ul_df"), pd.DataFrame)], ignore_index=True)
    dl_all = pd.concat([r["iperf_dl_df"] for r in results if isinstance(r.get("iperf_dl_df"), pd.DataFrame)], ignore_index=True)

    def stats_dict(prefix: str, s: pd.Series) -> Dict[str, float]:
        d = summarize_series(s)
        out = {
            f"{prefix}_median": d.get("median", np.nan),
            f"{prefix}_p25": d.get("p25", np.nan),
            f"{prefix}_p75": d.get("p75", np.nan),
            f"{prefix}_p95": d.get("p95", np.nan),
            f"{prefix}_mean": d.get("mean", np.nan),
        }
        return out

    # OAI: concatenate across tests
    ss_rsrp = pd.concat([r["oai_series"]["ss_rsrp_dbm"] for r in results], ignore_index=True)
    snr = pd.concat([r["oai_series"]["snr_db"] for r in results], ignore_index=True)
    dl_bler = pd.concat([r["oai_series"]["dl_bler_pct"] for r in results], ignore_index=True)
    ul_bler = pd.concat([r["oai_series"]["ul_bler_pct"] for r in results], ignore_index=True)

    row = {"scenario": label}

    row.update(stats_dict("SS_RSRP_dBm", ss_rsrp))
    row.update(stats_dict("SNR_dB", snr))
    row.update(stats_dict("DL_BLER_pct", dl_bler))
    row.update(stats_dict("UL_BLER_pct", ul_bler))

    if not dl_all.empty:
        row.update(stats_dict("DL_throughput_Mbps", dl_all["bitrate_Mbps"]))
        row.update(stats_dict("DL_jitter_ms", dl_all["jitter_ms"]))
        row.update(stats_dict("DL_loss_pct", dl_all["loss_percent"]))

    if not ul_all.empty:
        row.update(stats_dict("UL_throughput_Mbps", ul_all["bitrate_Mbps"]))
        row.update(stats_dict("UL_jitter_ms", ul_all["jitter_ms"]))
        row.update(stats_dict("UL_loss_pct", ul_all["loss_percent"]))

    return pd.DataFrame([row])
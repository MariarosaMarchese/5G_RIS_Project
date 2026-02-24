import os
import pandas as pd

from analysis_functions import (
    ensure_dir,
    find_tests,
    analyze_one_test,
    aggregate_distance,
    build_final_kpi_table,
    plot_cdf,
    plot_time_series,
    plot_cdf_overlay,
    plot_boxplot_two_groups
)

# -----------------------------
# Configuration
# -----------------------------
# Assumes you're running from: 5G_RIS_Project/analysis/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "testing_new")
FIG_DIR = os.path.join(PROJECT_ROOT, "analysis", "figures")
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "outputs")

ensure_dir(FIG_DIR)
ensure_dir(OUT_DIR)

# Define what "steady-state" means
SKIP_FIRST_S = 10.0   # remove first 10s
SKIP_LAST_S = 0.0     # keep full tail; set to e.g. 5.0 if you want

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_DIR:", DATA_DIR)


def main():
    tests = find_tests(DATA_DIR)
    if not tests:
        raise RuntimeError(
            "No tests found. Check that your folder is: data/testing_new/testing/{1m,2m}/testX/"
        )

    # Group results by distance
    results_by_dist = {"1m": [], "2m": []}

    for dist, test_name, test_path in tests:
        print(f"Analyzing {dist}/{test_name} ...")
        r = analyze_one_test(test_path, skip_first_s=SKIP_FIRST_S, skip_last_s=SKIP_LAST_S)
        r["dist"] = dist
        r["test_name"] = test_name
        results_by_dist[dist].append(r)

        # Per-test plots (optional but useful for debugging)
        if not r["iperf_ul_df"].empty:
            plot_time_series(
                r["iperf_ul_df"],
                ycol="jitter_ms",
                title=f"{dist} {test_name} UL jitter (steady-state)",
                ylabel="Jitter (ms)",
                outpath=os.path.join(FIG_DIR, f"{dist}_{test_name}_UL_jitter_timeseries.png"),
            )
            plot_time_series(
                r["iperf_ul_df"],
                ycol="bitrate_Mbps",
                title=f"{dist} {test_name} UL throughput (steady-state)",
                ylabel="Throughput (Mbps)",
                outpath=os.path.join(FIG_DIR, f"{dist}_{test_name}_UL_thr_timeseries.png"),
            )

        if not r["iperf_dl_df"].empty:
            plot_time_series(
                r["iperf_dl_df"],
                ycol="jitter_ms",
                title=f"{dist} {test_name} DL jitter (steady-state)",
                ylabel="Jitter (ms)",
                outpath=os.path.join(FIG_DIR, f"{dist}_{test_name}_DL_jitter_timeseries.png"),
            )
            plot_time_series(
                r["iperf_dl_df"],
                ycol="bitrate_Mbps",
                title=f"{dist} {test_name} DL throughput (steady-state)",
                ylabel="Throughput (Mbps)",
                outpath=os.path.join(FIG_DIR, f"{dist}_{test_name}_DL_thr_timeseries.png"),
            )

    # Build summary tables for each distance
    final_tables = []
    for dist in ["1m", "2m"]:
        dist_results = results_by_dist[dist]
        if not dist_results:
            continue

        # Per-test summary tables (UL/DL)
        ul_test_table = aggregate_distance(dist_results, direction="ul")
        dl_test_table = aggregate_distance(dist_results, direction="dl")

        ul_test_table.to_csv(os.path.join(OUT_DIR, f"{dist}_UL_per_test_summary.csv"), index=False)
        dl_test_table.to_csv(os.path.join(OUT_DIR, f"{dist}_DL_per_test_summary.csv"), index=False)

        # Final KPI row for thesis tables
        kpi_table = build_final_kpi_table(dist_results, label=dist)
        kpi_table.to_csv(os.path.join(OUT_DIR, f"{dist}_KPI_table.csv"), index=False)
        final_tables.append(kpi_table)

        # CDF plots across all tests (good for professor: jitter distribution)
        ul_all = pd.concat([r["iperf_ul_df"] for r in dist_results], ignore_index=True)
        dl_all = pd.concat([r["iperf_dl_df"] for r in dist_results], ignore_index=True)

        if not ul_all.empty:
            plot_cdf(
                ul_all["jitter_ms"],
                title=f"{dist} UL jitter CDF (steady-state)",
                xlabel="Jitter (ms)",
                outpath=os.path.join(FIG_DIR, f"{dist}_UL_jitter_CDF.png"),
            )
            plot_cdf(
                ul_all["bitrate_Mbps"],
                title=f"{dist} UL throughput CDF (steady-state)",
                xlabel="Throughput (Mbps)",
                outpath=os.path.join(FIG_DIR, f"{dist}_UL_thr_CDF.png"),
            )

        if not dl_all.empty:
            plot_cdf(
                dl_all["jitter_ms"],
                title=f"{dist} DL jitter CDF (steady-state)",
                xlabel="Jitter (ms)",
                outpath=os.path.join(FIG_DIR, f"{dist}_DL_jitter_CDF.png"),
            )
            plot_cdf(
                dl_all["bitrate_Mbps"],
                title=f"{dist} DL throughput CDF (steady-state)",
                xlabel="Throughput (Mbps)",
                outpath=os.path.join(FIG_DIR, f"{dist}_DL_thr_CDF.png"),
            )

        # -----------------------------------------
    # Overlay CDF plots: 1m vs 2m (same figure)
    # -----------------------------------------
    if results_by_dist["1m"] and results_by_dist["2m"]:
        ul_1m = pd.concat([r["iperf_ul_df"] for r in results_by_dist["1m"]], ignore_index=True)
        dl_1m = pd.concat([r["iperf_dl_df"] for r in results_by_dist["1m"]], ignore_index=True)

        ul_2m = pd.concat([r["iperf_ul_df"] for r in results_by_dist["2m"]], ignore_index=True)
        dl_2m = pd.concat([r["iperf_dl_df"] for r in results_by_dist["2m"]], ignore_index=True)

        # DL jitter overlay
        if not dl_1m.empty and not dl_2m.empty:
            plot_cdf_overlay(
                dl_1m["jitter_ms"], dl_2m["jitter_ms"],
                label1="1m", label2="2m",
                title="DL jitter CDF (steady-state): 1m vs 2m",
                xlabel="Jitter (ms)",
                outpath=os.path.join(FIG_DIR, "DL_jitter_CDF_1m_vs_2m.png"),
                xlog=False
            )

            plot_cdf_overlay(
                dl_1m["bitrate_Mbps"], dl_2m["bitrate_Mbps"],
                label1="1m", label2="2m",
                title="DL throughput CDF (steady-state): 1m vs 2m",
                xlabel="Throughput (Mbps)",
                outpath=os.path.join(FIG_DIR, "DL_thr_CDF_1m_vs_2m.png"),
                xlog=False
            )

        # UL jitter + throughput overlay (log scale recommended)
        if not ul_1m.empty and not ul_2m.empty:
            plot_cdf_overlay(
                ul_1m["jitter_ms"], ul_2m["jitter_ms"],
                label1="1m", label2="2m",
                title="UL jitter CDF (steady-state): 1m vs 2m",
                xlabel="Jitter (ms)",
                outpath=os.path.join(FIG_DIR, "UL_jitter_CDF_1m_vs_2m.png"),
                xlog=True
            )

            plot_cdf_overlay(
                ul_1m["bitrate_Mbps"], ul_2m["bitrate_Mbps"],
                label1="1m", label2="2m",
                title="UL throughput CDF (steady-state): 1m vs 2m",
                xlabel="Throughput (Mbps)",
                outpath=os.path.join(FIG_DIR, "UL_thr_CDF_1m_vs_2m.png"),
                xlog=True
            )
        
                # -----------------------------------------
        # Boxplots for UL (better visualization)
        # -----------------------------------------
        if not ul_1m.empty and not ul_2m.empty:

            # UL jitter boxplot
            plot_boxplot_two_groups(
                ul_1m["jitter_ms"],
                ul_2m["jitter_ms"],
                label1="1m",
                label2="2m",
                title="UL Jitter Comparison (steady-state)",
                ylabel="Jitter (ms)",
                outpath=os.path.join(FIG_DIR, "UL_jitter_boxplot_1m_vs_2m.png"),
                ylog=True   # very important due to 2–400 ms range
            )

            # UL throughput boxplot
            plot_boxplot_two_groups(
                ul_1m["bitrate_Mbps"],
                ul_2m["bitrate_Mbps"],
                label1="1m",
                label2="2m",
                title="UL Throughput Comparison (steady-state)",
                ylabel="Throughput (Mbps)",
                outpath=os.path.join(FIG_DIR, "UL_thr_boxplot_1m_vs_2m.png"),
                ylog=True   # important due to 0.02 vs 3 Mbps
            )

    # Combined KPI table (1m vs 2m)
    if final_tables:
        combined = pd.concat(final_tables, ignore_index=True)
        combined.to_csv(os.path.join(OUT_DIR, "KPI_table_1m_vs_2m.csv"), index=False)
        print("\nSaved combined KPI table to:", os.path.join(OUT_DIR, "KPI_table_1m_vs_2m.csv"))
        print(combined.to_string(index=False))

    print("\nDone. Outputs in:")
    print("  -", OUT_DIR)
    print("  -", FIG_DIR)


if __name__ == "__main__":
    main()
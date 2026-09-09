import pandas as pd
import argparse
import sys

# Candidate names for the chromosome column. The two input CSVs may use
# different conventions (e.g. 'Chromosome' in the CNV csv vs 'Chr' in the
# XCNV output), so we auto-detect and normalize instead of requiring the
# caller to pass matching key names.
CHROM_CANDIDATES = ['Chromosome', 'chromosome', 'Chr', 'chr', 'chrom', 'CHROM']
# Coordinate columns (besides the chromosome column) used as join keys.
COORD_KEYS = ['Start', 'End', 'Type']


def detect_chrom_col(df, label):
    for c in CHROM_CANDIDATES:
        if c in df.columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract the LR score column from the XCNV CSV and merge it "
                    "into the CNV CSV. Column-name differences (e.g. 'Chromosome' "
                    "vs 'Chr') are auto-detected, so no extra key arguments are "
                    "needed when calling through add_LR.sh.")
    parser.add_argument('--file1', type=str, required=True,
                        help="Original CNV CSV (the same one passed to annotate.sh)")
    parser.add_argument('--file2', type=str, required=True,
                        help="XCNV LR-score CSV (https://github.com/kbvstmd/XCNV)")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to save the merged CSV file")
    parser.add_argument('--lr_column', type=str, default='LR_pred',
                        help="Name of the LR-score column in file2 (default: LR_pred)")
    args = parser.parse_args()

    df1 = pd.read_csv(args.file1)
    df2 = pd.read_csv(args.file2)

    # --- auto-detect & normalize the chromosome column ---
    chrom1 = detect_chrom_col(df1, 'file1')
    chrom2 = detect_chrom_col(df2, 'file2')
    if chrom1 is None:
        sys.exit(f"[Error] file1 has no chromosome column (looked for {CHROM_CANDIDATES}).\n"
                 f"  file1 columns: {list(df1.columns)}")
    if chrom2 is None:
        sys.exit(f"[Error] file2 (XCNV) has no chromosome column (looked for {CHROM_CANDIDATES}).\n"
                 f"  file2 columns: {list(df2.columns)}")
    if chrom2 != chrom1:
        print(f"[Info] normalizing file2 chromosome column '{chrom2}' -> '{chrom1}' for join")
        df2 = df2.rename(columns={chrom2: chrom1})

    # --- auto-detect the LR column ---
    if args.lr_column not in df2.columns:
        sys.exit(f"[Error] LR column '{args.lr_column}' not found in file2.\n"
                 f"  file2 columns: {list(df2.columns)}")

    # --- derive join keys: chromosome + common coordinate columns ---
    keys = [chrom1] + [k for k in COORD_KEYS if k in df1.columns and k in df2.columns]
    if not keys:
        # Fallback: use the intersection of all columns (excluding the LR column).
        common = [c for c in df1.columns if c in df2.columns and c != args.lr_column]
        if not common:
            sys.exit("[Error] file1 and file2 share no common join-key columns.")
        keys = common

    print(f"[Info] file1 (CNV) rows={len(df1)}, file2 (XCNV) rows={len(df2)}, "
          f"join keys={keys}")

    # --- de-duplicate file2 on the join keys (keep first) ---
    # XCNV outputs can contain repeated CNV coordinates; a plain left-join would
    # inflate file1's rows and break the 1:1 alignment with the h5 predictions
    # produced in step 1 (annotate.sh). Dropping duplicates keeps the merged CSV
    # at exactly file1's row count and order.
    before = len(df2)
    df2 = df2.drop_duplicates(subset=keys, keep='first')
    dropped = before - len(df2)
    if dropped:
        print(f"[Info] dropped {dropped} duplicate rows from file2 on keys {keys} "
              f"(kept first occurrence per coordinate)")

    df_merged = pd.merge(
        df1,
        df2[keys + [args.lr_column]],
        on=keys,
        how='left',
    )

    # A left join on unique keys must preserve file1's row count and order.
    if len(df_merged) != len(df1):
        sys.exit(
            f"[Error] merge changed row count ({len(df1)} -> {len(df_merged)}). "
            f"This usually means file1 ({args.file1}) is NOT the same CNV set "
            f"used in step 1 (annotate.sh). The h5 predictions were generated "
            f"from a different number of rows, so the LR column cannot be aligned. "
            f"Re-run with the exact CNV csv that was passed to annotate.sh.")

    # Safety: warn loudly (do NOT silently proceed) if any row failed to match.
    # Common causes: a chr-prefix mismatch in the values (e.g. '1' vs 'chr1'),
    # or the two files covering different CNV sets.
    n_nan = int(df_merged[args.lr_column].isna().sum())
    if n_nan:
        print(f"[Warning] {n_nan}/{len(df_merged)} rows have NO matching LR score "
              f"(column '{args.lr_column}'). Check that the key values (chr prefix, "
              f"Type) and the CNV sets match between the two files. These rows "
              f"will receive NaN LR scores.")

    df_merged.to_csv(args.output_file, index=False)
    print(f"[Info] merged CSV written to {args.output_file} "
          f"({len(df_merged)} rows; join keys={keys})")


if __name__ == "__main__":
    main()

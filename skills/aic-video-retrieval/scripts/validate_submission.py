#!/usr/bin/env python3
"""
validate_submission.py - Submission validator for AIC 2026 video retrieval.
Validates submission CSV format, column counts, row counts (<= 100), and TRAKE strictly monotonic frame conditions.
"""

import argparse
import sys
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="AIC 2026 Submission File Validator")
    parser.add_argument("--file", type=str, required=True,
                        help="Path to CSV submission file to validate")
    parser.add_argument("--type", choices=["kis", "qa", "trake"], required=True,
                        help="Query type: 'kis', 'qa', or 'trake'")
    parser.add_argument("--max-rows", type=int, default=100,
                        help="Maximum allowed submission rows (default: 100)")
    return parser.parse_args()


def validate_kis_row(row_idx, row):
    errors = []
    if len(row) != 2:
        errors.append(f"Row {row_idx}: Expected exactly 2 columns (<video_id>, <frame_id>), got {len(row)}: {row}")
        return errors

    video_id, frame_str = row[0].strip(), row[1].strip()
    if not video_id:
        errors.append(f"Row {row_idx}: video_id is empty")

    try:
        frame_idx = int(frame_str)
        if frame_idx < 0:
            errors.append(f"Row {row_idx}: frame_id must be non-negative, got {frame_idx}")
    except ValueError:
        errors.append(f"Row {row_idx}: frame_id '{frame_str}' is not a valid integer")

    return errors


def validate_qa_row(row_idx, row):
    errors = []
    if len(row) < 3:
        errors.append(f"Row {row_idx}: Expected at least 3 columns (<video_id>, <frame_id>, <answer>), got {len(row)}: {row}")
        return errors

    video_id, frame_str = row[0].strip(), row[1].strip()
    # Join the rest as answer in case answer contains commas
    answer = ",".join(row[2:]).strip()

    if not video_id:
        errors.append(f"Row {row_idx}: video_id is empty")

    try:
        frame_idx = int(frame_str)
        if frame_idx < 0:
            errors.append(f"Row {row_idx}: frame_id must be non-negative, got {frame_idx}")
    except ValueError:
        errors.append(f"Row {row_idx}: frame_id '{frame_str}' is not a valid integer")

    if not answer:
        errors.append(f"Row {row_idx}: answer is empty")

    return errors


def validate_trake_row(row_idx, row):
    errors = []
    if len(row) < 3:
        errors.append(f"Row {row_idx}: Expected at least 3 columns (<video_id>, <frame_1>, <frame_2>, ...), got {len(row)}: {row}")
        return errors

    video_id = row[0].strip()
    if not video_id:
        errors.append(f"Row {row_idx}: video_id is empty")

    frames = []
    for col_idx, f_str in enumerate(row[1:], 1):
        f_str = f_str.strip()
        try:
            f_val = int(f_str)
            if f_val < 0:
                errors.append(f"Row {row_idx}, Event {col_idx}: frame_id must be non-negative, got {f_val}")
            frames.append(f_val)
        except ValueError:
            errors.append(f"Row {row_idx}, Event {col_idx}: '{f_str}' is not a valid integer")

    # CRITICAL: Check strictly increasing sequence: f_1 < f_2 < ... < f_n
    if len(frames) == len(row[1:]):
        for j in range(len(frames) - 1):
            if frames[j] >= frames[j + 1]:
                errors.append(
                    f"Row {row_idx} [CRITICAL TRAKE VIOLATION]: Frames must be strictly increasing! "
                    f"Event {j+1} (frame {frames[j]}) >= Event {j+2} (frame {frames[j+1]})"
                )

    return errors


def main():
    args = parse_args()
    file_path = Path(args.file)

    print("\n" + "=" * 60)
    print("📋 AIC 2026 SUBMISSION VALIDATION REPORT")
    print("=" * 60)
    print(f"• File Path        : {file_path}")
    print(f"• Submission Type  : {args.type.upper()}")
    print(f"• Max Allowed Rows : {args.max_rows}")
    print("-" * 60)

    if not file_path.exists():
        print(f"❌ ERROR: File does not exist: {file_path}", file=sys.stderr)
        sys.exit(1)

    all_errors = []
    row_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader, 1):
            if not row or all(not cell.strip() for cell in row):
                # Skip empty lines
                continue
            row_count += 1

            if args.type == "kis":
                errs = validate_kis_row(row_count, row)
            elif args.type == "qa":
                errs = validate_qa_row(row_count, row)
            elif args.type == "trake":
                errs = validate_trake_row(row_count, row)
            else:
                errs = [f"Unsupported query type '{args.type}'"]

            all_errors.extend(errs)

    if row_count == 0:
        all_errors.append("File contains 0 non-empty submission rows.")
    elif row_count > args.max_rows:
        all_errors.append(f"Exceeded max allowed rows: {row_count} rows found (limit: {args.max_rows}).")

    print(f"• Total Rows Checked: {row_count}")

    if all_errors:
        print(f"❌ STATUS: FAILED ({len(all_errors)} errors detected):\n")
        for err in all_errors[:20]:
            print(f"   [!] {err}")
        if len(all_errors) > 20:
            print(f"   ... and {len(all_errors) - 20} more errors.")
        print("=" * 60 + "\n")
        sys.exit(1)
    else:
        print("✅ STATUS: PASSED! All submission rules satisfied.")
        print("=" * 60 + "\n")
        sys.exit(0)


if __name__ == "__main__":
    main()

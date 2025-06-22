import argparse
import csv
import sys
from pathlib import Path
from statistics import mean
import subprocess


def _parse_csv(path: Path) -> float:
    """Return the average of the 'sm' column in an nvidia-smi dmon csv file."""
    # If the file doesn't exist or is empty, the run was too short to capture data.
    if not path.exists() or path.stat().st_size == 0:
        return 0.0

    sm_values = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = None
        sm_idx = -1
        for row in reader:
            # Skip comment or empty lines (e.g. # Date, Time, ...)
            if not row or row[0].strip().startswith("#"):
                continue

            if header is None:
                header = [col.strip().lower() for col in row]
                try:
                    sm_idx = header.index("sm")
                except ValueError:
                    # If header is malformed or missing 'sm', we can't get data.
                    # This can happen if dmon writes an error. Treat as 0 util.
                    return 0.0
                continue

            # Data row
            try:
                value = float(row[sm_idx])
                sm_values.append(value)
            except (ValueError, IndexError):
                # Ignore malformed data rows.
                continue

    if not sm_values:
        # Also handles case where there's a header but no data rows.
        return 0.0

    return mean(sm_values)


def _capinfos_total_bytes(pcap_path: Path) -> int:
    """Get total data bytes from a pcap file using capinfos."""
    try:
        # Use -B to get the total data size of all packets in bytes.
        result = subprocess.run(
            ["capinfos", "-B", str(pcap_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        # Output is just a number, e.g., "12345"
        return int(result.stdout.strip())
    except FileNotFoundError:
        print(f"WARNING: 'capinfos' tool not found. Install with: sudo apt-get install wireshark-common", file=sys.stderr)
        print(f"Network byte measurement will be reported as 0 for this run.", file=sys.stderr)
        return 0
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        print(f"WARNING: capinfos failed on {pcap_path}: {e}", file=sys.stderr)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute average SM utilisation from nvidia-smi dmon CSV.")
    parser.add_argument("csv_file", type=Path, help="Path to dmon csv output file")
    args = parser.parse_args(argv)

    # Wrap in a try/except to guarantee a successful exit code, which prevents
    # the parent experiment driver from crashing on any unexpected parsing error.
    try:
        avg = _parse_csv(args.csv_file)
        print(f"AVG_SM_UTIL: {avg:.2f}%")
    except Exception as exc:
        # Print the actual error to stderr for debugging, but print a default
        # '0.00%' to stdout and exit cleanly.
        print(f"Error parsing {args.csv_file}: {exc}", file=sys.stderr)
        print("AVG_SM_UTIL: 0.00%")


if __name__ == "__main__":
    main() 
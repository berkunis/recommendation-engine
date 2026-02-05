import numpy as np
from scipy import stats


def print_section_header(title):
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title):
    width = 50
    print(f"\n--- {title} " + "-" * max(0, width - len(title) - 5))


def print_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def print_distribution_summary(data, name):
    print(f"\n  {name} Distribution Summary:")
    print(f"    Count:      {len(data)}")
    print(f"    Mean:       {np.mean(data):.2f}")
    print(f"    Median:     {np.median(data):.2f}")
    print(f"    Std Dev:    {np.std(data):.2f}")
    print(f"    Skewness:   {stats.skew(data):.2f}")
    print(f"    Min:        {np.min(data):.2f}")
    print(f"    Max:        {np.max(data):.2f}")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"    P{p:<2}:        {np.percentile(data, p):.2f}")


def print_bar(label, value, max_value, width=40):
    bar_len = int((value / max(max_value, 1)) * width)
    bar = "█" * bar_len + "░" * (width - bar_len)
    print(f"  {label:>12s} |{bar}| {value}")

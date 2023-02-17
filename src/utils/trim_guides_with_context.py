import os
import sys
import pandas as pd


def main() -> int:
    path = sys.argv[1]
    guide_col = sys.argv[2]

    df = pd.read_csv(path)
    output_name = os.path.join(*path.split('/')[:-1], 'trimmed_' + path.split('/')[-1])

    df['guide'] = [g[2:22] for g in df[guide_col]]
    
    # Reorder
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    df.to_csv(output_name, index=False)

    return 0


if __name__ == '__main__':
    sys.exit(main())
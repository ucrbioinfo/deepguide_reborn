import sys
from get_context import find_guide_context

def main() -> int:
    fasta_file = sys.argv[1]

    find_guide_context(
        input_genome_path=fasta_file,
        nt_toward_3_prime=5,
        nt_toward_5_prime=2,
        spacer_length=20,
        output=True
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
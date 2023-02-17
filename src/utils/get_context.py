# You have a genome as input (.fasta/.fa)
# You have a guide RNA library with 20nt sequences
# You would like to find the nucleotide context around
# these 20nt guides from the genome.
# You use this script.
import re
import os
import sys
import yaml
import pandas
import argparse
from Bio import SeqIO
from Bio.Seq import Seq


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', 
        type=argparse.FileType(mode='r'),
        default='src/utils/get_context_config.yaml', 
        help='The config file to use.',
    )

    # Read the values in config.yaml
    args = parser.parse_args()
    arg_dict = vars(args)
    if args.config:
        arg_dict.update(yaml.load(args.config, Loader=yaml.FullLoader))

    return args


# NGG with lookahead for cas9 - # lookahead also captures overlaps
def find_guide_context(
    input_genome_path: str,
    nt_toward_3_prime: int, 
    nt_toward_5_prime: int, 
    spacer_length: int, 
    pam_regex_to_look_for: str = r'(?=(GG))', 
    output: bool = False
    ) -> dict:
     
    records = list(SeqIO.parse(input_genome_path, 'fasta'))
    
    # String all chromosomes together
    sequence = str()
    for record in records:
        sequence += str(record.seq).upper()

    # Store reverse complement
    sequence_rev_comp = str(Seq(sequence).reverse_complement())

    # Entries are of the form:
    # 'guide': [CONTEXTguideCONTEXT, CONTEXTguideCONTEXT, ...]
    guide_context_dict = dict() 

    # Forward strand
    matches = re.finditer(pam_regex_to_look_for, sequence)
    pam_positions = [match.start() for match in matches]

    # by default we used 2 nucleotides after the 20nt guide. Used 21 for indexing, then 2 more
    nt_toward_5_prime = (21 + nt_toward_5_prime)

    # + 1 to account for the fact that finditer() finds the index of where the PAM starts
    # which is stored in position below
    # If we have sequence=ACTAGG then position for GG is 4. We don't want the N in AGG which is A
    # If we want ACT (guide length 3), we take 4
    # and do sequence[position-(spacer_length+1):position-1] or sequence[0:3] and get ACT.
    guide_start_position = spacer_length + 1

    for position in pam_positions:
        if (position - nt_toward_5_prime >= 0) and (position + nt_toward_3_prime < len(sequence)):
            guide = sequence[position-guide_start_position:position-1]
            guide_with_context = sequence[position-nt_toward_5_prime:position+nt_toward_3_prime]

            if guide in guide_context_dict:
                guide_context_dict[guide].append(guide_with_context)
            else:
                guide_context_dict[guide] = [guide_with_context]


    # Reverse complementary strand
    matches = re.finditer(pam_regex_to_look_for, sequence_rev_comp)
    pam_positions = [match.start() for match in matches]
    
    for position in pam_positions:
        if (position - nt_toward_5_prime >= 0) and (position + nt_toward_3_prime < len(sequence_rev_comp)):
            guide = sequence_rev_comp[position-guide_start_position:position-1]
            guide_with_context = sequence_rev_comp[position-nt_toward_5_prime:position+nt_toward_3_prime]
            
            if guide in guide_context_dict:
                guide_context_dict[guide].append(guide_with_context)
            else:
                guide_context_dict[guide] = [guide_with_context]

    # Check which/how many guides have different contexts (same spacer sequence, 
    # different context around it).
    def all_equal(iterator) -> bool:
        iterator = iter(iterator)
        try:
            first = next(iterator)
        except StopIteration:
            return True
        return all(first == x for x in iterator)

    guide_single_context_dict = dict()
    diff_contexts = dict()  # Guides with the same spacer, different context around them
    for key, value in guide_context_dict.items():
        guide_single_context_dict[key] = value[0]

        if not all_equal(value):  # Checks to see if all guides with the same spacer also have the same context - if not, save them for inspection
            diff_contexts[key] = value

    print('{n1}/{n2} guides ({percent:.2f}%) in the genome have different contexts. Selecting the first context'.format(
        n1=len(diff_contexts),
        n2=len(guide_context_dict),
        percent=len(diff_contexts) / len(guide_context_dict) * 100,
    ))

    if output:
        pandas.DataFrame(list(diff_contexts.items()), columns=['guide', 'guide_with_context']).to_csv(
            'guides_with_diff_context.csv', index=False
        )

    if output:
        print('See output in {name}'.format(name='guides_with_context.csv'))
        pandas.DataFrame(
            list(guide_single_context_dict.items()),
            columns=['guide', 'guide_with_context']
        ).to_csv(
            'guides_with_context.csv', index=False,
        )
    
    return guide_context_dict


def get_librarys_context_from_genome(
    library_path: str,
     guides_col_name: str,
     guides_and_context_from_genome_dict: dict,
     output: bool = False,
    ) -> pandas.DataFrame:

    library = pandas.read_csv(library_path)

    context_list = list()  # List of guides with context around them. 2bp upstream, 3 pam, 2 downstream
    guides_not_found_in_genome_list = list()  # Tuples of (index_in_lib, sequence)
    
    library[guides_col_name] = library[guides_col_name].str.upper()

    for index, row in library.iterrows():
        if row[guides_col_name] in guides_and_context_from_genome_dict:
            context_list.append(guides_and_context_from_genome_dict[row[guides_col_name]][0])
        else:
            context_list.append('Not found in genome')
            guides_not_found_in_genome_list.append((index, row[guides_col_name]))

    library['guide_with_context'] = context_list

    print('{n1}/{n2} guides in the library ({percent:.2f}%) were not found in the genome.'.format(
        n1=len(guides_not_found_in_genome_list),
        n2=len(library),
        percent=len(guides_not_found_in_genome_list) / len(library),
    ))

    if output:
        with open('guides_not_found_in_genome.txt', 'w') as f:
            for i in guides_not_found_in_genome_list:
                f.write(str(i))
                f.write('\n')

    return library


def main() -> int:
    args = parse_arguments()

    input_genome_path = os.path.join(args.input_path, args.input_genome_name)
    input_library_path = os.path.join(args.input_path, args.input_library_name)

    output_file_name = args.input_library_name.split('.csv')[0] + '_context.csv'
    output_library_path = os.path.join(args.output_path, output_file_name)

    guides_and_context_from_genome_dict = find_guide_context(
        input_genome_path=input_genome_path, 
        nt_toward_3_prime=args.nt_toward_3_prime,
        nt_toward_5_prime=args.nt_toward_5_prime,
        spacer_length=args.spacer_length,
        pam_regex_to_look_for=args.pam_regex_to_look_for,
        output=args.output_context
    )

    library_df = get_librarys_context_from_genome(
        library_path=input_library_path,
        guides_col_name=args.input_guide_col_name,
        guides_and_context_from_genome_dict=guides_and_context_from_genome_dict,
        output=args.output_context
    )

    library_df = library_df.drop(
        library_df[library_df['guide_with_context'].isin(
            ['Not found in genome']
            )].index
        ).reset_index(drop=True)

    library_df.to_csv(output_library_path, index=False)
    print('Library output in {path}'.format(path=output_library_path))

    return 0


if __name__ == '__main__':
    sys.exit(main())

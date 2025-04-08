import numpy as np
import pandas as pd
from Bio import SeqIO
import random

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def extract_alt_sequence(genome_file, excel_file, output_fa_file):
    """
    Sequences were extracted from genome files and Excel files.

    parameter:
    genome_file (str): Genome file name.
    excel_file (str): Excel file name.
    sequence_length_needed (int): The length of the desired sequence is 98304 by default.

    return:
    list: List of extracted sequences.
    """
    genome = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))

    df = pd.read_csv(excel_file)

    alt_sequences = []

    # Calculate new start and end positions and acquire sequences.
    for index, row in df.iterrows():
        chrom = str(row['Chromosome'])
        start = int(row['Start'])
        end = int(row['End'])
        variant_type = row['Type']
        end_start_length = int(row['end-start'])
        sequence_length_needed = 98304

        # if chrom in genome:
        #     chromosome_seq = genome[chrom].seq

        if "chr" + chrom in genome:
            chromosome_seq = genome["chr" + chrom].seq
            original_sequence = chromosome_seq[start:end + 1]  # Extract original sequence

            if variant_type == 'gain':
                # Gain type, sequence after replication
                repeated_sequence = original_sequence + original_sequence
                total_length = len(repeated_sequence)
                extra_length_needed = sequence_length_needed - total_length
                if extra_length_needed > 0:
                    left_extra = extra_length_needed // 2
                    right_extra = extra_length_needed - left_extra
                    new_start = max(start - left_extra, 0)
                    left_extra_overflow = left_extra - (start - new_start)
                    new_end = min(end + len(original_sequence) + right_extra, len(chromosome_seq))
                    right_extra_overflow = right_extra - (new_end - (end + len(original_sequence)))
                    # Handle overflow on the left
                    if left_extra_overflow > 0:
                        new_end = new_end + left_extra_overflow
                    # Handle overflow on the right
                    if right_extra_overflow > 0:
                        new_start = new_start - right_extra_overflow - 1
                elif extra_length_needed < 0:
                    new_start = end - sequence_length_needed // 2
                    new_end = end + sequence_length_needed // 2 - 1
                    if new_start < 0:
                        new_start = 0
                        new_end = end + sequence_length_needed // 2 + (
                                sequence_length_needed // 2 - len(original_sequence))
                    if new_end > len(chromosome_seq):
                        new_start = end - sequence_length_needed // 2 - (
                                end + sequence_length_needed // 2 - len(chromosome_seq))
                        new_end = len(chromosome_seq)

                else:
                    new_start = start
                    new_end = end + len(original_sequence)
                new_sequence = chromosome_seq[new_start:new_end + 1]
                alt_sequences.append(str(new_sequence))

            elif variant_type == 'loss':
                # Loss type, ignoring the original sequence, and adding half on both sides
                half = sequence_length_needed // 2
                new_start = max(start - half, 0)
                left_extra_overflow = half - (start - new_start)
                new_end = min(end + half, len(chromosome_seq))
                right_extra_overflow = half - (new_end - end)
                # Handle overflow on the left
                if left_extra_overflow > 0:
                    new_end = min(new_end + left_extra_overflow, len(chromosome_seq))
                # Handle overflow on the right
                if right_extra_overflow > 0:
                    new_start = max(new_start - right_extra_overflow, 0)
                left_sequence = chromosome_seq[new_start:start]
                right_sequence = chromosome_seq[end:new_end]
                new_sequence = left_sequence + right_sequence
                alt_sequences.append(str(new_sequence))

        # Save the extracted sequence to a.Fa file
    with open(output_fa_file, "w") as output_handle:
        for i, seq in enumerate(alt_sequences):
            print(f"Sequence {i} length: {len(seq)}")
            seq_record = SeqRecord(Seq(seq), id=f"sequence_{i + 1}", description="")
            SeqIO.write(seq_record, output_handle, "fasta")

    print(f"Total sequences written: {len(alt_sequences)}")


extract_alt_sequence("/mnt/data0/users/baiy/CNV/resources/hg38_UCSC.fa",
                     "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3.csv",
                     "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3_alt_sequence.fa")


# print(len(extracted_sequences))


def extract_ref_sequence(genome_file, excel_file, output_fa_file):
    """
    Sequences were extracted from genome files and Excel files.

    parameter:
    genome_file (str): Genome file name.
    excel_file (str): Excel file name.
    sequence_length_needed (int): The length of the desired sequence is 98304 by default.

    return:
    list: List of extracted sequences.
    """

    genome = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))


    df = pd.read_csv(excel_file)

    ref_sequences = []

    # Calculate new start and end positions and acquire sequences
    for index, row in df.iterrows():
        chrom = str(row['Chromosome'])
        start = int(row['Start'])
        end = int(row['End'])
        variant_type = row['Type']
        end_start_length = int(row['end-start'])
        sequence_length_needed = 98304

        # if chrom in genome:
        #     chromosome_seq = genome[chrom].seq

        if "chr" + chrom in genome:
            chromosome_seq = genome["chr" + chrom].seq
            original_sequence = chromosome_seq[start:end + 1]

            mid_pos = start + ((end - start) // 2)

            new_start = mid_pos - sequence_length_needed // 2
            new_end = mid_pos + sequence_length_needed // 2
            # Handle overflow on the left
            if new_start < 0:
                new_start = 0
                new_end = mid_pos + sequence_length_needed // 2 + (sequence_length_needed // 2 - mid_pos)
            # Handle overflow on the right
            if new_end > len(chromosome_seq):
                new_end = len(chromosome_seq)
                new_start = mid_pos - sequence_length_needed // 2 - (
                            mid_pos + sequence_length_needed // 2 - len(chromosome_seq))
            new_sequence = chromosome_seq[new_start:new_end]
            ref_sequences.append(str(new_sequence))

    # # Save the extracted sequence to a.Fa file
    with open(output_fa_file, "w") as output_handle:
        for i, seq in enumerate(ref_sequences):
            print(f"Sequence {i} length: {len(seq)}")
            seq_record = SeqRecord(Seq(seq), id=f"sequence_{i + 1}", description="")
            SeqIO.write(seq_record, output_handle, "fasta")

    print(f"Total sequences written: {len(ref_sequences)}")


extract_ref_sequence("/mnt/data0/users/baiy/CNV/resources/hg38_UCSC.fa",
                     "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3.csv",
                     "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3_ref_sequence.fa")

print("Extract Finished")

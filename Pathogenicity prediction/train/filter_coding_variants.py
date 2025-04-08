import pandas as pd
from pybedtools import BedTool

def filter_noncoding_cnv(cnv_df, annotation_path):
    # 1. Load comment file
    if isinstance(annotation_path, str):
        annot_bed = BedTool(annotation_path)
    else:
        annot_bed = annotation_path

    # 2. The structural variation data were converted to bedtool format,
    #    and the first three columns were extracted as coordinate information
    cnv_bed = BedTool.from_dataframe(cnv_df[['Chromosome', 'Start', 'End']])

    # 3. Obtain variants that intersect with coding regions
    cnv_coding = cnv_bed.intersect(annot_bed, u=True).to_dataframe(names=['chrom', 'start', 'end'])

    print(cnv_coding.columns)

    # 4. Create a unique identifier column, add chrome, Start and end are combined into a string
    cnv_df['identifier'] = cnv_df['Chromosome'] + ':' + cnv_df['Start'].astype(str) + '-' + cnv_df['End'].astype(str)
    cnv_coding['identifier'] = cnv_coding['chrom'] + ':' + cnv_coding['start'].astype(str) + '-' + cnv_coding['end'].astype(str)

    # 5. If no variants were within the coding region, all were marked as noncoding
    if cnv_coding.shape[0] == 0:
        cnv_df['annotation'] = ['noncoding'] * cnv_df.shape[0]
    else:
        # 6. Marked as coding or noncoding according to the intersection results
        cnv_df['annotation'] = ['coding' if identifier in cnv_coding['identifier'].tolist() else 'noncoding'
                               for identifier in cnv_df['identifier']]

    # 7. Filtered out variants in noncoding regions
    cnv_noncoding = cnv_df[cnv_df['annotation'] == 'noncoding'].reset_index(drop=True)

    # 8. Return results, only data in the noncoding region
    return cnv_noncoding

cnv_df = pd.read_csv("/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/RecCNV_output_978.csv")

# Make sure the chrome column starts with "chr"
cnv_df['Chromosome'] = cnv_df['Chromosome'].apply(lambda x: 'chr' + str(x))

# Call the annotation function to load the annotation file
noncoding_cnv_df = filter_noncoding_cnv(cnv_df, '/mnt/data0/users/baiy/CNV/download_CNV_data/gencode_download/hg38_exon_gencode.bed')

# Save results to a new CSV file
noncoding_cnv_df.to_csv('/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/RecCNV_output_noncoding_variants.csv', index=False)

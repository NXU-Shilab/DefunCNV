# **DefunCNV**

Welcome to the `DefunCNV` framework repository! `DefunCNV` is a novel deep learning framework that enables precise annotation of brain related cis-regulatory elements and accurately predict the pathogenicity of functional noncoding CNVs in brain disorders.

# **Requirements**

Please create a new conda environment specifically for running `DefunCNV` (e.g. `conda create --DefunCNV python=3.8.20`), install the packages listed in the `requirements.txt` file. Install with conda or pip (e.g. `conda install pandas==2.0.3`).

# **Setup**

Please click [here](https://doi.org/10.5281/zenodo.22396868) to download and extract the trained `DefunCNV` model and  human reference genome files (containing hg19 and hg38 FASTA files) before proceeding:

# **Feature annotation**

The following scripts can be used to obtain feature annotations of DefunCNV for CNVs.

Example usage:

`sh annotate.sh <csv_file> <ref_fasta> <output_dir> [--cuda] `

Arguments:

*   `  <csv_file>  `: CSV file.
*   `  <ref_fasta>  `: Either hg19 or hg38 human reference genome.
*   `  <output_dir>  `: Path to feature annotation output directory.
*   `  --cuda  `: Optional, use this flag if running on a CUDA-enabled GPU.

The `  alt_sequence_predictions.h5  `and the `  ref_sequence_predictions.h5  `will be saved to output\_dir. There are 552 columns in the h5 file, and each column represents the predicted probability of a feature on each variant

# **CNVs pathogenicity prediction**

You should first use XCNV (<https://github.com/kbvstmd/XCNV>) to get LR score.Then the following scripts can be used to obtain DefunCNV pathogenicity predictions of for CNVs.

Example usage:

`sh add_LR.sh <ref_annotation> <alt_annotation> <csv_file1> <csv_file2> <output_dir> `

Arguments:

*   `<ref_annotation> `: ref sequence prediction h5 file.
*   `<alt_annotation> `: alt sequence prediction h5 file.
*   `<csv_file1> `: CSV file.
*   `<csv_file2> `: XCNV prediction CSV file.
*   `<output_dir> `: H5 file path with log2 score of LR.

The log2 scores with LR will be saved to output\_dir.

`sh predict.sh <h5_file> <best_model_path> <best_threshold> <output_dir> `

Arguments:

*   `h5_file> `: H5 file path with log2 score of LR.
*   `<best_model_path> `: Path to the DefunCNV bset model.
*   `<best_threshold> `: Path to the DefunCNV bset threshold.
*   `<output_dir> `: Path to pathogenicity predictions of for CNVs output directory.

The output will be saved to output\_dir, the first column of the output is the predicted class, the second column is the predicted score, and the last column is the best\_threshold.

# **Training**

### **Feature annotation model training**

The configuration file and script for running train is under the `Feature annotation/train` directory. To run DefunCNV deep learning feature annotation model training, you will need GPU computing capability (we run training on 6x NVIDIA GeForce 3090 GPUs).

The training data is available [here](https://drive.google.com/file/d/1NnO1VtgHGLagnzAPMa4b2_MAW1PyGnJB/view?usp=drive_link) should be downloaded and extracted into the `train` directory.

The DefunCNV feature annotation model training configuration YAML file is provided as the `train/train.yml` file. You can read more about the Selene command-line interface and configuration file formatting [here](https://selene.flatironinstitute.org/master/overview/cli.html#).

You must use Selene version >0.5.0 to train this model ([release notes](https://github.com/FunctionLab/selene/blob/master/RELEASE_NOTES.md)).

We also provide an example SLURM script `train.sh` for submitting a training job to a cluster.

### **Pathogenicity prediction model training**

After annotating the features, train a Random Forest model.We provide model training and parameter search code.Make sure to update the training dataset path in both `coding_noncoding_CNV_dataset_train.py` and `noncoding_CNV_dataset_train.py` with your own path.

Example usage of training on coding/noncoding dataset:

`python coding_noncoding_CNV_dataset_train.py`&#x20;

Arguments:

python your\_script\_name.py

Example usage of training on noncoding dataset:

`python noncoding_CNV_dataset_train.py`

Arguments:

python your\_script\_name.py.

# **Help**

Please post in the Github issues or e-mail Fangyuan Shi (<shify@nxu.edu.cn>) with any questions about the repository, requests for more data, additional information about the results, etc.

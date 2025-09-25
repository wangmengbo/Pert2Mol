# Run baseline
```bash
bash script/data_paths_gdp.sh; python diff_train_pert2mol.py --prefix gdp-diff --dataset gdp --drug-data-path $DRUG_DATA_PATH --raw-drug-csv-path $RAW_DRUG_CSV_PATH --image-json-path $IMAGE_JSON_PATH --gene-count-matrix-path $GENE_COUNT_MATRIX_PATH
```
# Download and untar
wget https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz \
--directory-prefix ./data/criteo/raw/

tar -xvzf ./data/criteo/raw/dac.tar.gz \
--directory ./data/criteo/raw/

# Split
python ./src/utils/criteo_split.py criteo-random-split \
--source_file_name="./data/criteo/raw/train.txt" \
--train_ratio=0.80 \
--train_file_name="./data/criteo/csv/train.csv" \
--test_file_name="./data/criteo/csv/test.csv"

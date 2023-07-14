mkdir -p data
mkdir -p misc

# download and process dataset
cd dataset_proc
# gsm
mkdir -p raw_data

wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

mv train.jsonl raw_data/gsm_train.jsonl
mv test.jsonl raw_data/gsm_test.jsonl

python proc_gsm.py

# ecqa 
wget https://storage.googleapis.com/feb-data/data.zip
unzip data.zip
mv data/ECQA-Dataset/ecqa_train.jsonl raw_data/ecqa_train.jsonl
mv data/ECQA-Dataset/ecqa_test.jsonl raw_data/ecqa_test.jsonl
rm -rf data
rm data.zip

python proc_ecqa.py


# esnli
wget https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_train_1.csv
wget https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_train_2.csv
wget https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv

mv esnli_train_1.csv raw_data/esnli_train_1.csv
mv esnli_train_2.csv raw_data/esnli_train_2.csv
mv esnli_test.csv raw_data/esnli_test.csv

python proc_esnli.py

# code 002 cached results
unzip code002cache.zip
cp 002cache/* ../misc
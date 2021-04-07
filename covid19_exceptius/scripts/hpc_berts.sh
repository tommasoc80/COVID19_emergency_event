#!/bin/bash
#SBATCH --job-name="covid19-berts"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module purge
module load Python
cd ~/COVID19_emergency_event

echo '======uk-mbert======='
python -m covid19_exceptius.scripts.train_bert -n 'mbert' -p './annotations/uk/annotations_uk_train.tsv' -tst './annotations/uk/annotations_uk_dev.tsv' -d 'cuda' -bs 16 -e 20 -s '/data/s3913171/COVID-19-event' -wd 0.02 
echo '======full-mbert======='
python -m covid19_exceptius.scripts.train_bert -n 'mbert' -p './annotations/annotations_full_multilingual_1.tsv' -tst './annotations/uk/annotations_uk_dev.tsv' -d 'cuda' -bs 16 -e 20 -s '/data/s3913171/COVID-19-event' -wd 0.02 
echo '======full-eng-legal=======' 
python -m covid19_exceptius.scripts.train_bert -n 'eng-legal' -p './annotations/annotations_full_eng_1.tsv' -tst './annotations/uk/annotations_uk_dev.tsv' -d 'cuda' -bs 16 -e 20 -s '/data/s3913171/COVID-19-event' -wd 0.02 

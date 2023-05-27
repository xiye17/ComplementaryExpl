# ComplementaryExpl
Code for the MMR exemplar-selection strategy described in the paper [Complementary Explanations for Effective In-Context Learning](https://arxiv.org/abs/2211.13892) (ACL Findings, 2023).

## Setup
* python==3.8
* requirements: pip install -r requirements.txt
* Set OPENAI KEY: export KEY=yourkey


## Experiments
Preparation:
`sh prepare_data.sh`


Example selection exeperiments on GSM:

`sh exp_scripts/gsm.sh`

Example selection exeperiments  ECQA:

`sh exp_scripts/ecqa.sh`

Example selection exeperiments  ESNLI:

`sh exp_scripts/esnli.sh`

## Citation
```
@InProceedings{Ye-Et-Al:2023:Complementary,
  title = {Complementary Explanations for Effective In-Context Learning},
  author = {Xi Ye and Srinivasan Iyer and Asli Celikyilmaz and Ves Stoyanov and Greg Durrett and Ramakanth Pasunuru},
  booktitle = {Findings of ACL},
  year = {2023},
}
```

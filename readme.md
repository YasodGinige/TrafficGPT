## TrafficGPT: An LLM Approach for Open-Set Encrypted Traffic Classification

![](https://img.shields.io/badge/license-MIT-000000.svg)
[![arXiv](https://img.shields.io/badge/arXiv-1909.05658-<color>.svg)]()

**Note:**
- ⭐ **Please leave a <font color='orange'>STAR</font> if you like this project!** ⭐
- If you find any <font color='red'>incorrect</font> / <font color='red'>inappropriate</font> / <font color='red'>outdated</font> content, please kindly consider opening an issue or a PR.

<div align="center">
    <img src="/CVA.png" width="850" height="450" alt="overall architecure"/>
</div>

In this repository, we guide you in setting up the TrafficGPT project in a local environment and reproducing the results. TrafficGPT, a novel traffic analysis attack that leverages GPT-2, a popular LLM, to enhance feature extraction, thereby improving
the open-set performance of downstream classification. We use five existing encrypted traffic datasets to show how the feature extraction by GPT-2 improves the open-set performance of traffic
analysis attacks. As the open-set classification methods, we use K-LND, OpenMax, and Backgroundclass methods, and shows that K-LND methods have higher performance overall.

**Datasets:** [AWF](https://arxiv.org/abs/1708.06376), [DF](https://arxiv.org/abs/1801.02265), [DC](https://www.semanticscholar.org/paper/Deep-Content%3A-Unveiling-Video-Streaming-Content-Li-Huang/f9feff95bc1d68674d5db426053f417bd2c8786b), [USTC](https://github.com/yungshenglu/USTC-TFC2016), [CSTNet-tls](https://drive.google.com/drive/folders/1JSsYmevkxQFanoKOi_i1ooA6pH3s9sDr)

**Openset methods**
- [K-LND methods](https://github.com/ThiliniDahanayaka/Open-Set-Traffic-Classification)
- OpenMax
- Background class

# Using TrafficGPT

First, clone the git repo and install the requirements.
```
git clone https://github.com/YasodGinige/TrafficGPT.git
cd TrafficGPT
pip install -r requirements.txt
```
Next, download the dataset and place it in the data directory.
```
gdown https://drive.google.com/uc?id=1-MVfxyHdQeUguBmYrIIw1jhMVSqxXQgO
unzip data.zip 
```

Then, preprocess the dataset you want to train and evaluate. Here, the dataset name should be DF, AWF, DC, USTC, or CSTNet.
```
python3 data_preprocess.py --data_path ./data --dataset <dataset_name>
```
To train the model, run the suitable code for the dataset:
```
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 60  --dataset DF
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 200  --dataset AWF
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 4  --dataset DC
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 12  --dataset USTC
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 75  --dataset CSTNet
```

To evaluate, run the suitable code for the dataset:
```
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 60 --K_number 30 --TH_value 0.8 --dataset DF
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 200 --K_number 50 --TH_value 0.9 --dataset AWF
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 4 --K_number 4 --TH_value 0.9 --dataset DC
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 12 --K_number 5 --TH_value 0.8 --dataset USTC
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 5 --num_labels 75 --K_number 20 --TH_value 0.8 --dataset CSTNe
```

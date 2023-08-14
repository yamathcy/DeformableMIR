# Deformable Convolution for Singing Technique Classification

Based on WandB and PyTorch Lightning

Modified data preprocessing to [MARBLE Benchmark](https://github.com/a43992899/MARBLE-Benchmark)

Preparation
1. Download VocalSet from https://zenodo.org/record/1193957
2. run the code
   
```
python prepare.py
```
Then, split files will be created.

Main program
```
python main.py
```


Reference:
```
@inproceedings{yamamoto22_interspeech,
  author={Yuya Yamamoto and Juhan Nam and Hiroko Terasawa},
  title={{Deformable CNN and Imbalance-Aware Feature Learning for Singing Technique Classification}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={2778--2782},
  doi={10.21437/Interspeech.2022-11137}
}
```

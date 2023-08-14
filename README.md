# Deformable Convolution for Singing Technique Classification

Based on WandB and PyTorch Lightning

Modified data preprocessing to [MARBLE Benchmark](https://github.com/a43992899/MARBLE-Benchmark)

Preparation
1. Download VocalSet from https://zenodo.org/record/1193957
2. Run the code
   
```
python prepare.py
```
Then, split files will be created.

Main program
```
python main.py
```

The detailed experiment conditions are contained in run.sh.

- APSIPA 2021: Refer to the line "OblongCNN"
- Interspeech 2022: Refer to the line "OblongDeformableCNN"
- The Best Version: Refer to the line "dcnv2 crt stft", which uses Multi-resolution spectrogram, classifier retraining, and modulated deformable convolution (DCNv2)


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

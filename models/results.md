**Results**

| Method | Feature Extractor | Dataset Augmentation | Model File | Details | Train Acc | Train Loss | Test Acc | Test Loss |
| :---: |:---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | N/A | N/A | N/A | Class distribution | N/A | N/A | **0.660** | 1.000 |
| Linear NN | AlexNet | None | alexnet_linearnn_2019-11-08_16:52:48.pth | 3 epochs | 0.704 | 0.766 | 0.591 | 1.086 |
| Linear NN | AlexNet | None | alexnet_linearnn_2019-11-08_16:54:04.pth | 5 epochs | 0.736 | 0.715 | 0.556 | 1.140 |
| Linear NN | AlexNet | None | alexnet_linearnn_2019-11-08_17:03:09.pth | 10 epochs | 0.788 | 0.568 | 0.567 | 1.227 |
| Linear NN | AlexNet | Over Sampling | alexnet_linearnn_2019-11-08_17:04:58.pth | 3 epochs | 0.663 | 0.780 | 0.393 | 1.513 |
| Linear NN | AlexNet | Over Sampling | alexnet_linearnn_2019-11-08_17:07:33.pth | 5 epochs | 0.739 | 0.620 | 0.435 | 1.586 |
| Linear NN | AlexNet | Over Sampling | alexnet_linearnn_2019-11-08_17:13:07.pth | 10 epochs | **0.802** | **0.469** | 0.460 | 1.881 |
| Bigger NN | AlexNet | None | alexnet_biggernn_2019-11-11_20:07:44.pth | 3 epochs | 0.657 | 0.944 | **0.660** | **0.960** |
| Bigger NN | AlexNet | None | alexnet_biggernn_2019-11-11_20:18:26.pth | 5 epochs | 0.657 | 0.929 | **0.660** | 0.967 |
| Bigger NN | AlexNet | Over Sampling | alexnet_biggernn_2019-11-11_20:32:35.pth | 3 epochs | 0.558 | 0.783 | 0.410 | 1.299 |
| LDA | AlexNet | Under Sampling | alexnet_lda_2019-11-11_19:43:01.pkl | N/A | 0.224 | 1.681 | 0.189 | 1.716 |
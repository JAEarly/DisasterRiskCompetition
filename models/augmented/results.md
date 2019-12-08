**Results**
  
| Method             |   Feature Extractor  | Train Acc | Train LL  |  Val Acc  |   Val LL  |  Test Acc |  Test LL  |   Cmp LL   |
| :----------------: | :------------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :--------: |
| LinearNN           | ResNetCustom         |   0.837   |   0.426   |   0.773   | **0.594** | **0.796** | **0.542** |     -      |
| BiggerNN           | ResNetCustom         | **0.845** | **0.400** | **0.776** |   0.596   |   0.777   |   0.553   |     -      |
| Retrained          | ResNet               |   0.771   |   0.582   |   0.755   |   0.643   |   0.770   |   0.596   |     -      |

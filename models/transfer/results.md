**Results**
  
| Method             |   Feature Extractor   | Train Acc | Train LL  |  Val Acc  |   Val LL  |  Test Acc |  Test LL  |   Cmp LL   |
| :----------------: | :-------------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :--------: |
| LinearNN           | ResNetCustom          |   0.838   |   0.427   | **0.817** | **0.519** | **0.794** | **0.552** |     -      |
| XGB                | ResNetCustom          | **0.928** | **0.217** |   0.816   |   0.520   |   0.793   |   0.573   |     -      |
| Retrained (2)      | ResNetCustom          |   0.819   |   0.473   |   0.782   |   0.580   |   0.779   |   0.604   |     -      |
| Retrained          | ResNetCustom          |   0.781   |   0.530   |   0.760   |   0.593   |   0.752   |   0.622   |     -      |

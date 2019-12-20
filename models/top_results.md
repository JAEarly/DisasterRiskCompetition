**Results**
  
| Method                 |   Feature Extractor   | Train Acc | Train LL  |  Val Acc  |   Val LL  |  Test Acc |  Test LL  |  Cmp LL   |  CF Plot  |
| :--------------------: | :-------------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| XGB Ensemble All       | ResNetCustom          |   0.888   |   0.305   |   0.830   |   0.449   |   0.825   |   0.448   |   0.533   |     -     |
| XGB All                | ResNetCustom          |   0.891   |   0.306   | **0.829** | **0.459** | **0.825** | **0.450** |   0.537   |     -     |
| LinearNN Ensemble All  | ResNetCustom          |   0.878   |   0.355   |   0.812   |   0.519   |   0.808   |   0.503   |     -     |     -     |
| LinearNN All           | ResNetCustom          |   0.871   |   0.365   |   0.799   |   0.525   |   0.806   |   0.508   | **0.514** |    True   |
| LinearNN Ensemble      | ResNetCustom          |   0.874   |   0.362   |   0.806   |   0.526   |   0.800   |   0.519   |   0.516   |     -     |
| LinearNN               | ResNetCustom          |   0.876   |   0.354   |   0.801   |   0.534   |   0.802   |   0.524   |     -     |     -     |
| BiggerNN               | ResNetCustom          |   0.882   |   0.346   |   0.806   |   0.537   |   0.802   |   0.531   |     -     |    True   |
| SVM                    | ResNetCustom          |   0.882   |   0.317   |   0.809   |   0.552   |   0.804   |   0.536   |     -     |     -     |
| XGB Ensemble           | ResNetCustom          |   0.891   |   0.300   |   0.825   |   0.445   |   0.801   |   0.540   |     -     |    True   |
| XGB                    | ResNetCustom          | **0.896** | **0.288** |   0.803   |   0.550   |   0.804   |   0.550   |     -     |    True   |
| Retrained Ensemble     | ResNetCustom          |   0.808   |   0.506   |   0.800   |   0.511   |   0.799   |   0.543   |     -     |     -     |

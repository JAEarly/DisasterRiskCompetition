**Results**

| Method      | Feature Extractor  | Train Acc | Train LL  |  Val Acc  |   Val LL  |  Test Acc | Test Loss |
| :---------: | :----------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| LinearNN    | ResNetCustom       |   0.888   |   0.305   | **0.859** | **0.399** | **0.851** | **0.400** |
| Retrain CNN | ResNet             |   0.876   |   0.355   |   0.847   |   0.423   |   0.839   |   0.432   |
| LinearNN    | AlexNetCustom      |   0.876   |   0.334   |   0.849   |   0.441   |   0.842   |   0.433   |
| LinearNN    | AlexNetCustomSmote |   0.637   |   0.781   |   0.846   |   0.440   |   0.842   |   0.435   |
| BiggerNN    | AlexNetCustom      |   0.905   | **0.268** |   0.843   |   0.443   |   0.836   |   0.441   |
| Retrain CNN | AlexNet            |   0.877   |   0.360   |   0.836   |   0.469   |   0.827   |   0.467   |
| LinearNN    | AlexNet            |   0.872   |   0.357   |   0.821   |   0.517   |   0.793   |   0.552   |
| LinearNN    | AlexNetSmote       |   0.914   |   0.278   |   0.781   |   0.610   |   0.768   |   0.625   |
| XGB         | AlexNet            | **0.918** |   0.495   |   0.759   |   0.753   |   0.735   |   0.784   |
| Baseline    | N/A                |   0.657   |   1.014   |   0.656   |   1.015   |   0.660   |   1.005   |

| -- Old Results Below -- | --------- | --------- | --------- | --------- | --------- | --------- |
| LinearNN    | ResNet             |   0.657   |   0.963   |   0.656   |   0.963   |   0.660   |   0.953   |
| BiggerNN    | ResNet  |   0.657   |   0.961   |   0.656   |   0.961   |   0.660   |   0.950   |
| BiggerNN    | ResNetSmote  |   0.657   |   1.590   |   0.656   |   1.590   |   0.660   |   1.590   |
| XGB         | AlexNetSmote       |   0.212   |   1.610   |   0.222   |   1.608   |   0.207   |   1.609   |
**Results**

| Method      | Feature Extractor  | Train Acc | Train LL  |  Val Acc  |   Val LL  |  Test Acc | Test Loss |
| :---------: | :----------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| LinearNN    | ResNetCustom       |   0.888   |   0.305   | **0.859** |   0.399   | **0.851** | **0.400** |
| BiggerNN    | ResNetCustom       |   0.888   |   0.308   | **0.859** | **0.394** |   0.847   |   0.402   |
| Retrain CNN | ResNet             |   0.876   |   0.355   |   0.847   |   0.423   |   0.839   |   0.432   |
| LinearNN    | AlexNetCustom      |   0.876   |   0.334   |   0.849   |   0.441   |   0.842   |   0.433   |
| LinearNN    | AlexNetCustomSmote |   0.899   |   0.287   |   0.846   |   0.440   |   0.842   |   0.435   |
| BiggerNN    | ResNet             |   0.877   |   0.329   |   0.848   |   0.400   |   0.831   |   0.437   |
| BiggerNN    | AlexNetCustom      |   0.905   |   0.268   |   0.843   |   0.443   |   0.836   |   0.441   |
| LinearNN    | ResNet             |   0.835   |   0.438   |   0.844   |   0.431   |   0.820   |   0.465   |
| Retrain CNN | AlexNet            |   0.877   |   0.360   |   0.836   |   0.469   |   0.827   |   0.467   |
| BiggerNN    | AlexNet            |   0.889   |   0.307   |   0.836   |   0.465   |   0.823   |   0.484   |
| LinearNN    | ResNetCustomSmote  |   0.849   |   0.402   |   0.812   |   0.508   |   0.795   |   0.518   |
| LinearNN    | AlexNet            |   0.872   |   0.357   |   0.821   |   0.517   |   0.793   |   0.552   |
| XGB         | AlexNetCustom      |   0.896   |   0.396   |   0.812   |   0.580   |   0.810   |   0.578   |
| BiggerNN    | ResNetSmote        |   0.850   |   0.380   |   0.807   |   0.522   |   0.793   |   0.582   |
| BiggerNN    | AlexNetCustomSmote |   0.888   |   0.293   |   0.799   |   0.587   |   0.800   |   0.591   |
| LinearNN    | AlexNetSmote       |   0.843   |   0.416   |   0.781   |   0.610   |   0.768   |   0.625   |
| BiggerNN    | AlexNetSmote       |   0.838   |   0.388   |   0.776   |   0.597   |   0.770   |   0.631   |
| LinearNN    | ResNetSmote        |   0.796   |   0.536   |   0.768   |   0.581   |   0.759   |   0.638   |
| XGB         | AlexNetSmote       | **0.986** | **0.191** |   0.727   |   0.731   |   0.706   |   0.776   |
| XGB         | AlexNet            |   0.918   |   0.495   |   0.759   |   0.753   |   0.735   |   0.784   |
| Baseline    | N/A                |   0.657   |   1.014   |   0.656   |   1.015   |   0.660   |   1.005   |

**Results**

| Method         | Feature Extractor  | Train Acc | Train LL  |  Val Acc  |   Val LL  |  Test Acc | Test Loss |
| :------------: | :----------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| LinearNN Kfold | ResNetCustom       |   0.890   |   0.308   |   0.863   | **0.387** |   0.845   | **0.399** |
| LinearNN       | ResNetCustom       |   0.888   |   0.305   |   0.859   |   0.399   |   0.851   |   0.400   |
| BiggerNN       | ResNetCustom       |   0.888   |   0.308   |   0.859   |   0.394   |   0.847   |   0.402   |
| XGB            | ResNetCustom       |   0.942   |   0.166   |   0.860   |   0.403   | **0.852** |   0.402   |
| BiggerNN Kfold | ResNetCustom       |   0.888   |   0.310   | **0.865** |   0.389   |   0.851   |   0.406   |
| Retrain CNN    | ResNet             |   0.876   |   0.355   |   0.847   |   0.423   |   0.839   |   0.432   |
| LinearNN       | AlexNetCustom      |   0.876   |   0.334   |   0.849   |   0.441   |   0.842   |   0.433   |
| LinearNN       | AlexNetCustomSmote |   0.899   |   0.287   |   0.846   |   0.440   |   0.842   |   0.435   |
| BiggerNN       | ResNet             |   0.877   |   0.329   |   0.848   |   0.400   |   0.831   |   0.437   |
| BiggerNN       | AlexNetCustom      |   0.905   |   0.268   |   0.843   |   0.443   |   0.836   |   0.441   |
| LinearNN       | ResNet             |   0.835   |   0.438   |   0.844   |   0.431   |   0.820   |   0.465   |
| Retrain CNN    | AlexNet            |   0.877   |   0.360   |   0.836   |   0.469   |   0.827   |   0.467   |
| XGB            | AlexNetCustom      |   0.938   |   0.204   |   0.825   |   0.482   |   0.829   |   0.471   |
| XGB            | ResNetCustomSmote  |   0.945   |   0.162   |   0.832   |   0.456   |   0.819   |   0.479   |
| BiggerNN       | AlexNet            |   0.889   |   0.307   |   0.836   |   0.465   |   0.823   |   0.484   |
| XGB            | ResNet             |   0.946   |   0.207   |   0.832   |   0.471   |   0.806   |   0.491   |
| XGB            | AlexNetCustomSmote |   0.926   |   0.237   |   0.828   |   0.472   |   0.814   |   0.492   |
| LinearNN       | ResNetCustomSmote  |   0.849   |   0.402   |   0.812   |   0.508   |   0.795   |   0.518   |
| LinearNN       | VggNet             |   0.835   |   0.434   |   0.814   |   0.498   |   0.795   |   0.525   |
| BiggerNN       | ResNetCustomSmote  |   0.846   |   0.388   |   0.805   |   0.519   |   0.795   |   0.529   |
| LinearNN       | AlexNet            |   0.872   |   0.357   |   0.821   |   0.517   |   0.793   |   0.552   |
| XGB            | ResNetSmote        |   0.918   |   0.283   |   0.791   |   0.525   |   0.778   |   0.557   |
| XGB            | AlexNetSmote       | **0.965** | **0.154** |   0.793   |   0.560   |   0.774   |   0.582   |
| BiggerNN       | ResNetSmote        |   0.850   |   0.380   |   0.807   |   0.522   |   0.793   |   0.582   |
| BiggerNN       | AlexNetCustomSmote |   0.888   |   0.293   |   0.799   |   0.587   |   0.800   |   0.591   |
| XGB            | AlexNet            |   0.895   |   0.331   |   0.782   |   0.582   |   0.757   |   0.605   |
| LinearNN       | AlexNetSmote       |   0.843   |   0.416   |   0.781   |   0.610   |   0.768   |   0.625   |
| BiggerNN       | AlexNetSmote       |   0.838   |   0.388   |   0.776   |   0.597   |   0.770   |   0.631   |
| LinearNN       | ResNetSmote        |   0.796   |   0.536   |   0.768   |   0.581   |   0.759   |   0.638   |
| Baseline       | N/A                |   0.657   |   1.014   |   0.656   |   1.015   |   0.660   |   1.005   |

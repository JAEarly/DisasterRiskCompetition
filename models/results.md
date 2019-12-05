**Results**

| Method             | Feature Extractor  | Train Acc | Train LL  |  Val Acc  |   Val LL  |  Test Acc |  Test LL  |   Cmp LL   |
| :----------------: | :----------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :--------: |
| *LinearNN          | ResNetCustom       |   0.888   |   0.305   |   0.634   |   1.147   |   0.851   |   0.400   | **0.5510** |
| *BiggerNN          | ResNetCustom       |   0.888   |   0.308   |   0.634   |   1.129   |   0.847   |   0.402   |   0.5659   |
| *XGB               | ResNetCustom       |   0.942   |   0.166   |   0.611   |   1.262   |   0.852   |   0.402   |   0.5779   |
| *LinearNN          | AlexNetCustom      |   0.876   |   0.334   |   0.523   |   1.375   |   0.842   |   0.433   |      -     |
| *LinearNN          | AlexNetCustomSmote |   0.637   |   0.781   |   0.535   |   1.323   |   0.842   |   0.435   |      -     |
| *BiggerNN          | ResNet             |   0.877   |   0.329   |   0.592   |   1.047   |   0.831   |   0.437   |   0.5517   |
| *BiggerNN          | AlexNetCustom      |   0.905   |   0.268   |   0.528   |   1.336   |   0.836   |   0.441   |      -     |
| *LinearNN          | ResNet             |   0.835   |   0.438   |   0.531   |   1.222   |   0.820   |   0.465   |      -     |
| *XGB               | AlexNetCustom      |   0.938   |   0.204   |   0.482   |   1.601   |   0.829   |   0.471   |      -     |
| *XGB               | ResNetCustomSmote  |   0.927   |   0.222   |   0.706   |   0.962   |   0.819   |   0.479   |      -     |
| *BiggerNN          | AlexNet            |   0.889   |   0.307   |   0.505   |   1.421   |   0.823   |   0.484   |      -     |
| *XGB               | ResNet             |   0.946   |   0.207   |   0.508   |   1.535   |   0.806   |   0.491   |      -     |
| *XGB               | AlexNetCustomSmote |   0.976   |   0.110   |   0.574   |   1.245   |   0.814   |   0.492   |      -     |
| BiggerNN (3)       | ResNetCustomSmote2 |   0.831   |   0.401   |   0.768   |   0.562   |   0.786   |   0.505   |   0.6520   |
| LinearNN           | ResNetCustomSmote  |   0.848   |   0.402   |   0.739   |   0.736   |   0.795   |   0.518   |      -     |
| *LinearNN          | VggNet             |   0.835   |   0.434   |   0.494   |   1.345   |   0.795   |   0.525   |      -     |
| BiggerNN           | ResNetCustomSmote  |   0.855   |   0.376   |   0.724   |   0.841   |   0.795   |   0.529   |      -     |
| *LinearNN          | AlexNet            |   0.872   |   0.357   |   0.529   |   1.615   |   0.793   |   0.552   |      -     |
| *XGB               | ResNetSmote        |   0.966   |   0.177   |   0.596   |   0.998   |   0.778   |   0.557   |      -     |
| *BiggerNN          | ResNetSmote        |   0.927   |   0.216   |   0.645   |   1.230   |   0.793   |   0.582   |      -     |
| *XGB               | AlexNetSmote       |   0.965   |   0.154   |   0.445   |   1.654   |   0.774   |   0.582   |      -     |
| *BiggerNN          | AlexNetCustomSmote |   0.966   |   0.109   |   0.592   |   1.740   |   0.802   |   0.619   |      -     |
| *LinearNN          | AlexNetSmote       |   0.914   |   0.278   |   0.542   |   1.455   |   0.768   |   0.625   |      -     |
| *BiggerNN          | AlexNetSmote       |   0.927   |   0.207   |   0.551   |   1.624   |   0.770   |   0.631   |      -     |
| *LinearNN          | ResNetSmote        |   0.887   |   0.354   |   0.677   |   0.848   |   0.759   |   0.638   |      -     |
| *XGB               | AlexNet            |   0.809   |   0.582   |   0.355   |   1.697   |   0.736   |   0.719   |      -     |

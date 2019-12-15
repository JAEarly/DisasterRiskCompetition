
from matplotlib import pyplot as plt

train_losses = [0.3704454220094411, 0.4403359111527472, 0.5091782730384596, 0.5601178702602804, 0.6183590757466589, 0.6392147312477442, 0.6564890054848348, 0.676558945131489]
pseudo_losses = [0.37044542201460423, 0.30508899455620525, 0.33570680589225804, 0.36441144141148823, 0.3981213875703807, 0.40940039608026674, 0.41945151316759316, 0.4313765143317302]
validation_losses = [0.5608398914337158, 0.6886984705924988, 0.773935079574585, 0.8354789018630981, 0.9022316336631775, 0.9229602813720703, 0.9457039833068848, 0.96519535779953]

assert len(train_losses) == len(pseudo_losses) == len(validation_losses)

xs = range(len(train_losses))

fig, axis = plt.subplots(nrows=1, ncols=1)
axis.plot(xs, train_losses, label="Train")
axis.plot(xs, pseudo_losses, label="Pseudo")
axis.plot(xs, validation_losses, label="Validation")
axis.legend(loc='best')
axis.set_xlabel('Iterations')
axis.set_ylabel('Loss')
plt.show()

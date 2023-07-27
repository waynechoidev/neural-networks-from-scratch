# inputs = [(0.2, 1600), (1.0, 11000), (1.4, 23000), (1.6, 24000), (2.0, 30000),
#           (2.2, 31000), (2.7, 35000), (2.8, 38000), (3.2,
#                                                      40000), (3.3, 21000), (3.5, 45000),
#           (3.7, 46000), (4.0, 50000), (4.4, 49000), (5.0, 60000), (5.2, 62000)]

# Normalized input
inputs = [(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
          (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
          (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
          (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)]
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290,
           870, 1545, 1480, 1750, 1845, 1790, 1955]

w1 = 0.1
w2 = 0.2
b = 0.3
epochs = 4000
learning_rate = 0.1


def predict(i1, i2):
    return w1 * i1 + w2 * i2 + b


# train the network
for epoch in range(epochs):
    # feed Forward
    pred = [predict(i1, i2) for i1, i2 in inputs]

    # cost
    cost = sum([(p-t) ** 2 for p, t in zip(pred, targets)]) / len(targets)
    print(f"Epoch: {epoch}, Cost: {cost:.2f}")

    # back propagation
    errors_d = [2 * (p-t) for p, t in zip(pred, targets)]
    weight1_d = [e * i[0] for e, i in zip(errors_d, inputs)]
    weight2_d = [e * i[1] for e, i in zip(errors_d, inputs)]
    bias_d = [e * 1 for e in errors_d]
    w1 -= learning_rate * sum(weight1_d)/len(weight1_d)
    w2 -= learning_rate * sum(weight2_d)/len(weight2_d)
    b -= learning_rate * sum(bias_d)/len(bias_d)

print(f"Weight1: {w1:.2f}, Weight2: {w2:.2f}, Bias: {b:.2f},")

# test the network
test_inputs = [(0.1600, 0.1391), (0.5600, 0.3046),
               (0.7600, 0.8013), (0.9600, 0.3046), (0.1600, 0.7185)]
test_targets = [500, 850, 1650, 950, 1375]
pred = [predict(i1, i2) for i1, i2 in test_inputs]
for p, t in zip(pred, test_targets):
    print(f"target:{t}, pred:{p:.0f}")

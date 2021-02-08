from src import *
import matplotlib.pyplot as plt
import numpy as np
import time

train = 1
save = 1

# epochs = 7
# batch_size = 100
# learning_rate = 0.001
# alpha = 0.001
# scaling = 0.99 / 255
#
# loader = MnistLoader(scaling)
# x_train, t_train = loader.get_train_data()
# x_test, t_test = loader.get_test_data()

epochs = 100
batch_size = 10
learning_rate = 0.001
split_idx = 500
alpha = 0
# gamma = 0.1

loader = GestureLoader(1, ('Neck', 'Nose', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip',
                           'LHip', 'MidHip'
                           , 'LEye', 'REye'
                           ))
x_train, t_train = loader.get_train_data()
x_test = x_train[:split_idx, :]
x_train = x_train[split_idx:, :]
t_test = t_train[:split_idx]
t_train = t_train[split_idx:]

if train:
    # Exercise small

    # epochs = 1
    # batch_size = 2
    # learning_rate = 0.02
    #
    # x_train, t_train = np.array([[1, -2], [1, -2]]), np.array([[0], [0]])
    #
    # network = Network(
    #     FullyConnected(2, 2, w=np.array([[-1, 1], [0.67, -0.67]]), b=np.array([2, -3])),
    #     Sigmoid(),
    #     FullyConnected(2, 2, w=np.array([[1, -0.33], [1, 0.67]]), b=np.array([1, -4])),
    #     Sigmoid(),
    #     FullyConnected(2, 1, w=np.array([[.67], [-1.3]]), b=np.array([.5])),
    #     Sigmoid()
    # )

    # Exercise delivery_data

    # import pandas as pd
    #
    # # load the data from the Logistic Regression module
    # delivery_data = pd.read_csv("../ml_hci_ws20/03_logistic_regression/data/delivery_data.csv")
    # epochs = 2000
    # batch_size = 1
    # learning_rate = 0.01
    # alpha = 0
    # split_idx = 10
    # # data_size x input_size; data_size x class_size
    # x_train, t_train = delivery_data[["motivation", "distance"]].values, delivery_data["delivery?"].values
    # t_train = np.reshape(t_train, (-1, 1))
    #
    # x_test = x_train[:split_idx, :]
    # x_train = x_train[split_idx:, :]
    # t_test = t_train[:split_idx, :]
    # t_train = t_train[split_idx:, :]
    #
    # network = Network(
    #     FullyConnected(2, 2),
    #     Sigmoid(),
    #     FullyConnected(2, 2),
    #     Sigmoid(),
    #     FullyConnected(2, 1),
    #     Sigmoid()
    # )

    # MNIST

    # network = load_network('MNIST_accuracy_0.9427_confidence_0.8521_lr_0.001_a_0.001_epochs_7')

    # network = Network(
    #     FullyConnected(784, 100),
    #     Sigmoid(),
    #     FullyConnected(100, 50),
    #     Sigmoid(),
    #     FullyConnected(50, 10)
    # )

    # Gestures

    network = Network(
            FullyConnected(676, 300),
            Sigmoid(),
            FullyConnected(300, 100),
            Sigmoid(),
            FullyConnected(100, 2)
        )

    print("Training for " + str(epochs) + " epochs with " + str(batch_size) + " batches and a learning rate of " + str(
        learning_rate) + ":")

    t = time.time()
    loss = Trainer(
        network,
        cce_loss,
        AdamOptimizer(learning_rate, alpha)
    ).train(x_train, t_train, epochs, batch_size)

    print("Training done in " + str(time.time() - t) + " seconds with final Loss of " + str(loss[len(loss) - 1]))
    plt.plot(loss)
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.title("Batch-size: " + str(batch_size) + " Learning-rate: " + str(learning_rate))
    plt.show()
else:
    network = load_network('MNIST_accuracy_0.9094_confidence_0.6992')


avg_tps, avg_confidence, correct_wrong_predictions = network.test(x_test, t_test)

if save:
    save_network(network, 'G_accuracy_' + str(round(avg_tps, 4)) + '_confidence_' + str(round(avg_confidence, 4)) +
                 '_lr_' + str(learning_rate) + '_a_' + str(alpha) + '_epochs_' + str(epochs))

# for i in range(len(frames)):
#     frame = frames[i]
#     error = 0
#     for j in range(-check_range, check_range):
#         if i+j >= len(frames) or i+j < 0 or frames[i+j]-j == frame:
#             error += 1
#     error /= 2*check_range+1
#     if error >= 0.6:
#         frame_indices.append(frame/total_length)
#
# extract_frames("./data/demo_video.mp4", frame_indices)

print("Accuracy: " + str(avg_tps))
print("Confidence: " + str(avg_confidence))

np.set_printoptions(precision=4)
for i in range(len(correct_wrong_predictions[0])):
    correct = len(correct_wrong_predictions[0][i])
    wrong = len(correct_wrong_predictions[1][i])
    print("Class " + str(i) + "\n" + str(correct) + " correct and " + str(wrong) + " wrong predictions -> " +
          str(100 * wrong / (correct + wrong)) + " % wrong.")

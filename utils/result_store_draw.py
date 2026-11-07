import numpy as np
import matplotlib.pyplot as plt


def draw_result_visualization(folder, epoch_result):

    np.savetxt(folder + r"\\epoch.txt", epoch_result, fmt="%.4f", delimiter=',', newline='\n')
    plt.figure()
    plt.plot(epoch_result[:][0], epoch_result[:][1])
    plt.title("the change of the loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(folder + r"\\loss_change.png")
    plt.show()
    plt.figure()
    plt.plot(epoch_result[:][0], epoch_result[:][2])
    plt.title("the change of the accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig(folder + r"\\accuracy_change.png")



def store_result(folder, OA, AA, Kappa, CM, model_type, mode, epoch, batch_size, patches, band_patches, learning_rate, weight_decay, gamma, sample_mode, sample_value):
    with open(folder + r"\\accuracy.txt", 'w', encoding="utf-8") as f:
        f.write("Parameter settings:" + "\n")
        f.write("model_type : " + model_type + "\n")
        f.write("model_mode : " + mode + "\n")
        f.write("epoch : " + str(epoch) + "\n")
        f.write("batch_size : " + str(batch_size) + "\n")
        f.write("patches : " + str(patches) + "\n")
        if model_type == "Transformer":
            f.write("band_patches : " + str(band_patches) + "\n")
        f.write("learning_rate : " + str(learning_rate) + "\n")
        f.write("weight_decay : " + str(weight_decay) + "\n")
        f.write("gamma : " + str(gamma) + "\n")
        f.write("sample_mode : " + sample_mode + "\n")
        f.write("sample_value : " + str(sample_value) + "\n")
        f.write("Model result:" + "\n")
        f.write("OA : {:.4f}\n".format(OA))
        f.write("AA : {:.3f}\n".format(AA))
        f.write("Kappa : {:.3f}\n".format(Kappa))
        f.write("Confusion Matrix :\n")
        f.write("{}".format(CM))

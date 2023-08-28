
import numpy as np
import matplotlib.pyplot as plt

def plot_ec_curve(ec_path):
    ecs = np.loadtxt(ec_path)
    num_curves = ecs.shape[0]
    color_cycle = plt.cm.rainbow(np.linspace(0, 1, num_curves))
    for i in range(num_curves):
        plt.plot(ecs[i], color=color_cycle[i])

    plt.ylabel("EC")
    plt.title("SECT curves of 19 tumors from BRATS dataset")
    plt.show()



def main():
    plot_ec_curve("/Users/baihesun/cancer_data/BRATS_nifti_out/ECT.txt")

if __name__ == "__main__":
    main()
    print("done")


import matplotlib.pyplot as plt

def performance_plot(train_metric: list, 
                     valid_metric: list, 
                     outfile=None,
                     title: str = "",
                     xlab: str = "Epochs",
                     ylab:str="Metric",
                     save_plot:bool=True):
    
    fig, ax = plt.subplots()
    plt.plot(train_metric, color="b", alpha=0.7, marker='o', label="train", lw=3)
    plt.plot(valid_metric, color="orange", alpha=0.8, marker='o',  label="valid", lw=3)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_plot == False:
        plt.show()
    elif save_plot == True:
        if outfile == None:
            plt.savefig("metric_plot.png")
        else:
            plt.savefig(outfile)
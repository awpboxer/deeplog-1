import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from OpenStack import const
def plot_train_valid_loss(save_dir):
    train_loss = pd.read_csv(save_dir + "train_log.csv")
    valid_loss = pd.read_csv(save_dir + "valid_log.csv")
    sns.lineplot(x="epoch",y="loss" , data = train_loss, label="train loss")
    sns.lineplot(x="epoch",y="loss" , data = valid_loss, label="valid loss")
    plt.title("epoch vs train loss vs valid loss")
    plt.legend
    plt.savefig(save_dir+"train_valid_loss.png")
    plt.show()



if __name__ == "__main__":
    save_dir = const.OUTPUT_DIR + const.PARSER + "_result2/deeplog/"
    plot_train_valid_loss(save_dir)
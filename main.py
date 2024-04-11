
from model_training import train_and_plot

if __name__ == "__main__":
    dataset_number = "7"   # Change this to the desired dataset number
    train_and_plot(dataset_number, num_top_features=3)

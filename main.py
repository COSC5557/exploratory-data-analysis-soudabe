import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_frame = pd.read_csv("winequality-white.csv", sep=";")
    print(data_frame.columns)

    features = data_frame.drop('quality', axis=1).columns.tolist()

    data_frame[features].hist(figsize=(12, 8), bins=20)
    plt.suptitle('Feature Distributions Before Preprocessing')
    plt.tight_layout()
    plt.show()

    transformed_data = data_frame[features].apply(lambda x: np.log(x + 1))  # Adding 1 to avoid log(0)

    transformed_data.hist(figsize=(12, 8), bins=20)
    plt.suptitle('Feature Distributions After Preprocessing')
    plt.tight_layout()
    plt.show()


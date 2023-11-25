import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    data_frame = pd.read_csv("winequality-white.csv", sep=";")
    print(data_frame.columns)

    selected_features = ['fixed acidity', 'free sulfur dioxide', 'total sulfur dioxide']

    selected_data = data_frame[selected_features]

    selected_data.hist(figsize=(10, 6), bins=20)
    plt.suptitle('Feature Distributions Before Normalization')
    plt.tight_layout()
    plt.show()

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(selected_data)
    normalized_data = pd.DataFrame(normalized_data, columns=selected_features)

    normalized_data.hist(figsize=(10, 6), bins=20)
    plt.suptitle('Feature Distributions After Normalization')
    plt.tight_layout()
    plt.show()

import torch


# 3 layers MLP model
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 3 layers 1D CNN model
class CNN1D(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(CNN1D, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, hidden_dim, 3, padding="same")
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim, 3, padding="same")
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


# 3 layers LSTM model
class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(1))
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


if __name__ == "__main__":
    x = torch.randn(64, 1, 256)

    # Test MLP model
    model = MLP(256, 1840)
    y = model(x)
    print(model, y.shape)

    # Test CNN1D model
    model = CNN1D(1, 1130)
    y = model(x)
    print(model, y.shape)

    # Test LSTM model
    model = LSTM(1, 440)
    y = model(x)
    print(model, y.shape)

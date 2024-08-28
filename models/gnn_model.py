import torch
import torch.nn.functional as F
import os
from torch_geometric.nn import NNConv, GlobalAttention
from torch_geometric.loader import DataLoader
from torch.nn import Sequential as Seq, Linear, ReLU
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch


class GNN(torch.nn.Module):
    def __init__(self, input_size, edge_attr_size, global_feat_size):
        super(GNN, self).__init__()

        # Define a neural network to transform edge features
        nn1 = Seq(Linear(edge_attr_size, 32), ReLU(), Linear(32, input_size * 64))
        self.conv1 = NNConv(input_size, 64, nn1, aggr='mean')

        nn2 = Seq(Linear(edge_attr_size, 32), ReLU(), Linear(32, 64 * 32))
        self.conv2 = NNConv(64, 32, nn2, aggr='mean')

        # Global attention pooling (to incorporate global features)
        self.attention = GlobalAttention(gate_nn=Linear(32, 1))

        # Final prediction layers (including global features)
        self.fc1 = torch.nn.Linear(32 + global_feat_size, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, edge_attr, global_feat, batch):
        # First graph convolutional layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)

        # Second graph convolutional layer
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        # Global attention pooling
        x = self.attention(x, batch=batch)

        # Concatenate global features
        global_feat = global_feat.unsqueeze(1)

        x = torch.cat([x, global_feat], dim=1)

        # Final fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class GNNModel:
    def __init__(self, config):
        self.config = config
        self.model_dir = os.path.join(config['paths']['result'], config['experiment_name'])
        os.makedirs(self.model_dir, exist_ok=True)

        self.tensorboard_log_dir = os.path.join(self.model_dir, 'logs')
        self.checkpoints_dir = os.path.join(self.model_dir, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # Model is not built here
        self.optimizer = None
        self.writer = SummaryWriter(log_dir=self.tensorboard_log_dir)

    def build_model(self):
        input_size = 1  # Number of features per node
        edge_attr_size = 1  # Number of features per edge
        global_feat_size = 1  # Number of global features (e.g., previous cycle capacity)
        self.model = GNN(input_size=input_size, edge_attr_size=edge_attr_size, global_feat_size=global_feat_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config['model_specifications']['learning_rate'])

    def train_model(self, train_loader, val_loader):
        best_val_loss = float('inf')
        patience = self.config['model_specifications']['early_stopping_patience']
        epochs_without_improvement = 0

        for epoch in range(self.config['model_specifications']['epochs']):
            self.model.train()
            total_train_loss = 0

            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.global_feat, batch.batch)
                loss = F.mse_loss(out, batch.y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * batch.num_graphs

            avg_train_loss = total_train_loss / len(train_loader.dataset)

            # Validation phase
            val_loss = self.validate(val_loader)

            print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)

            # Check for improvement and save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, 'best_model.pth'))
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping due to lack of improvement in validation loss.")
                break

    def validate(self, val_loader):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.global_feat, batch.batch)
                loss = F.mse_loss(out, batch.y)
                total_val_loss += loss.item() * batch.num_graphs
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        return avg_val_loss

    def load_model(self, exp_path):
        if self.model is None:
            self.build_model()  # Build the model before loading
        self.model.load_state_dict(torch.load(os.path.join(exp_path, 'checkpoints', 'best_model.pth')))
        self.model = self.model.to(self.device)

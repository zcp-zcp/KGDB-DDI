import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader
import sys
import torch.nn.init as init
import os

sys.path.append("../..")
from Models.GAT import GAT
from DATASet import DataSet
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, \
    precision_score, auc, accuracy_score
import sklearn.metrics as m
import warnings
import datetime
from argparse import ArgumentParser
import numpy as np
from sklearn.utils import resample

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")

parser = ArgumentParser(description='Model Training.')
parser.add_argument('--batch_size', default=64, type=int,
                    metavar='N',
                    help='Set the mini-batch size for training. This controls the number of '
                         'samples processed in one forward/backward pass of the network. '
                         'Larger batch sizes may improve training speed but require more memory. '
                         'Default: 8.')
parser.add_argument('--dataset', type=str, default='Drugbank',
                    choices=['Drugbank', 'KEGG_DRUG'],
                    metavar='DATASET',
                    help='Select the dataset to use for training. Default: KEGG_DRUG.')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--gpu', type=int, default=0, help='the number of GPU')
parser.add_argument('--p', type=float, default=0.1, help='dropout')


args = parser.parse_args()

if args.dataset == 'KEGG_DRUG':
    set_seed(seed=6)
if args.dataset == 'Drugbank':
    set_seed(seed=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        new_dim = input_dim
        while new_dim % num_heads != 0:
            new_dim += 1

        self.head_dim = new_dim // num_heads

        self.q_linear = nn.Linear(input_dim, new_dim)
        self.v_linear = nn.Linear(input_dim, new_dim)
        self.k_linear = nn.Linear(input_dim, new_dim)
        self.out_linear = nn.Linear(new_dim, input_dim)

    def forward(self, input_vector):
        batch_size = input_vector.size(0)

        q = self.q_linear(input_vector).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_linear(input_vector).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_linear(input_vector).view(batch_size, -1, self.num_heads, self.head_dim)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        attention_weights = torch.nn.Softmax(dim=-1)(attention_scores)

        attended_output = torch.matmul(attention_weights, v)

        attended_output = attended_output.contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        output = self.out_linear(attended_output)
        output = output.squeeze(1)
        return output


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        new_dim = input_dim
        while new_dim % num_heads != 0:
            new_dim += 1

        self.head_dim = new_dim // num_heads

        self.q_linear = nn.Linear(input_dim, new_dim)
        self.v_linear = nn.Linear(input_dim, new_dim)
        self.k_linear = nn.Linear(input_dim, new_dim)
        self.out_linear = nn.Linear(new_dim, input_dim)

    def forward(self, A_features, B_features):
        batch_size = A_features.size(0)

        q = self.q_linear(B_features)
        k = self.v_linear(A_features)
        v = self.k_linear(A_features)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)

        scores = torch.matmul(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(self.head_dim))

        attention_weights = nn.Softmax(dim=-1)(scores)

        cross_attended = torch.matmul(attention_weights, v)
        cross_attended = cross_attended.contiguous().view(batch_size, -1, self.head_dim * self.num_heads)

        output = self.out_linear(cross_attended)
        output = output.squeeze(1)
        return output


class LayerNormClass(torch.nn.Module):
    def __init__(self, num_features):
        super(LayerNormClass, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(num_features)

    def forward(self, vector):
        return self.layer_norm(vector)


class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForward, self).__init__()
        self.layer1 = nn.Linear(input_dim, 1024)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x


class InteractionPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, K1, K2, dropout_p=args.p):
        super(InteractionPredictor, self).__init__()

        self.multi_head_attention = MultiHeadAttention(64, num_heads=K1)
        self.cross_attention = MultiHeadCrossAttention(128, num_heads=K2)
        self.layer_norm = LayerNormClass(256)
        self.ff_net = FeedForward(512, 256)

        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.fc4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(p=dropout_p)
        self.fc5 = nn.Linear(64, 32)
        # self.fc9 = nn.Linear(16, 8)
        # self.fc10 = nn.Linear(8, 4)
        self.dropout5 = nn.Dropout(p=dropout_p)
        self.fc6 = nn.Linear(32, 16)
        self.dropout6 = nn.Dropout(p=dropout_p)
        self.fc7 = nn.Linear(16, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.zeros_(m.bias)

    def forward(self, node1_features, node2_features, drug1_text_emb, drug2_text_emb):
        batch_size = node1_features.size(0)
        node1_features = node1_features.view(batch_size, -1)
        node2_features = node2_features.view(batch_size, -1)
        drug1_text_emb = drug1_text_emb.view(batch_size, -1)
        drug2_text_emb = drug2_text_emb.view(batch_size, -1)

        att_node1_features = self.multi_head_attention(node1_features)
        att_node2_features = self.multi_head_attention(node2_features)
        att_drug1_text_emb = self.multi_head_attention(drug1_text_emb)
        att_drug2_text_emb = self.multi_head_attention(drug2_text_emb)

        drug1_features = torch.cat((att_node1_features, att_drug1_text_emb), dim=1)
        drug2_features = torch.cat((att_node2_features, att_drug2_text_emb), dim=1)

        drug1_cross_features = self.cross_attention(drug1_features, drug2_features)
        drug2_cross_features = self.cross_attention(drug2_features, drug1_features)

        # combined_features = combined_features.reshape(batch_size, -1)
        concat_drug1_features = torch.cat((drug1_cross_features, drug1_features), dim=1)  # 256    128+128
        concat_drug2_features = torch.cat((drug2_cross_features, drug2_features), dim=1)

        # concat_features1 = torch.cat((drug1_cross_features, drug2_features), dim=1)  # 256    128+128
        # concat_features2 = torch.cat((drug2_cross_features, drug1_features), dim=1)

        norm_features1 = self.layer_norm(concat_drug1_features)  # 256
        norm_features2 = self.layer_norm(concat_drug2_features)

        combined_features = torch.cat((norm_features1, norm_features2), dim=1)  # 512

        new_features = self.ff_net(combined_features)

        norm_new_features = self.layer_norm(new_features)

        x = F.relu(self.fc1(norm_new_features))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        x = F.relu(self.fc6(x))
        x = self.dropout6(x)
        x = torch.sigmoid(self.fc7(x))

        return x


if torch.cuda.is_available():
    print("CUDA is available! Training on GPU ...")
    dev = f"cuda:{args.gpu}"
else:
    print("CUDA is not available! Training on CPU ...")
    dev = "cpu"

output, node_map, out_channels = GAT.all_drugs_fea(dataset=args.dataset)
node_feature_dim = out_channels

num_epochs = args.epochs

input_dim = 4 * node_feature_dim
output_dim = 1

train_file_path = ''
valid_file_path = ''
test_file_path = ''
K1 = 1
K2 = 1
if args.dataset == "Drugbank":
    train_file_path = '../../Data/Drugbank/train.txt'
    valid_file_path = '../../Data/Drugbank/valid.txt'
    test_file_path = '../../Data/Drugbank/test.txt'
    K1 = 6
    K2 = 1

if args.dataset == "KEGG_DRUG":
    train_file_path = '../../Data/KEGG_DRUG/train.txt'
    valid_file_path = '../../Data/KEGG_DRUG/valid.txt'
    test_file_path = '../../Data/KEGG_DRUG/test.txt'
    K1 = 1
    K2 = 1


model = InteractionPredictor(input_dim, output_dim, K1, K2)
device = torch.device(dev)
model.to(device)


criterion = nn.BCELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=args.lr)

train_dataset = DataSet.InteractionDataset(train_file_path, output, node_map, args.dataset)
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

valid_dataset = DataSet.InteractionDataset(valid_file_path, output, node_map, args.dataset)
valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

test_dataset = DataSet.InteractionDataset(test_file_path, output, node_map, args.dataset)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

def bootstrap_ci(y_true, y_pred, metric_func, B=2000):
    scores = []
    for _ in range(B):
        indices = resample(np.arange(len(y_true)), replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        scores.append(score)
    # print(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return (lower, upper)


def train(epoch, num_epochs, dataset):
    log_save_path = ''
    if dataset == 'Drugbank':
        log_save_path = '../../Log/Drugbank/drugbank_train_log.txt'
    if dataset == 'KEGG_DRUG':
        log_save_path = '../../Log/KEGG_DRUG/kegg_drug_train_log.txt'
    running_loss = 0.0
    iteration_counter = 0

    for node1_features, node2_features, drug1_text_emb, drug2_text_emb, labels in train_data_loader:
        labels = labels.unsqueeze(1)
        node1_features = node1_features.to(device)
        node2_features = node2_features.to(device)
        drug1_text_emb = drug1_text_emb.to(device)
        drug2_text_emb = drug2_text_emb.to(device)
        labels = labels.to(device)
        # data = data.to(torch.float32)
        node1_features = node1_features.to(torch.float32)
        node2_features = node2_features.to(torch.float32)
        drug1_text_emb = drug1_text_emb.to(torch.float32)
        drug2_text_emb = drug2_text_emb.to(torch.float32)
        outputs = model(node1_features, node2_features, drug1_text_emb, drug2_text_emb)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iteration_counter += 1
        if iteration_counter % 300 == 0:
            # avg_loss = running_loss / iteration_counter
            # print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss}')
            with open(log_save_path, 'a') as f:
                avg_loss = running_loss / iteration_counter
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f'Time,{current_time},Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss}')
                f.write("\n")
            running_loss = 0.0
            iteration_counter = 0
    # test()


def valid(epoch, num_epochs, dataset, best_val_accuracy):

    log_save_path = ''
    if dataset == 'Drugbank':
        log_save_path = '../../Log/Drugbank/HY_drugbank_valid_log.txt'
    if dataset == 'KEGG_DRUG':
        log_save_path = '../../Log/KEGG_DRUG/HY_kegg_drug_valid_log.txt'
    correct_predictions = 0
    total_predictions = 0
    all_true_labels = []
    all_outputs = []

    with torch.no_grad():
        for node1_features, node2_features, drug1_text_emb, drug2_text_emb, labels in valid_data_loader:
            node1_features = node1_features.to(device).to(torch.float32)
            node2_features = node2_features.to(device).to(torch.float32)
            drug1_text_emb = drug1_text_emb.to(device).to(torch.float32)
            drug2_text_emb = drug2_text_emb.to(device).to(torch.float32)
            labels = labels.to(device)

            outputs = model(node1_features, node2_features, drug1_text_emb, drug2_text_emb)
            predicted = (outputs.data > 0.5).float()

            total_predictions += labels.size(0)
            correct_predictions += (predicted.squeeze(1) == labels).sum().item()

            all_true_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.squeeze(1).cpu().numpy())

        all_true_labels = np.array(all_true_labels)
        all_outputs = np.array(all_outputs)
        all_predicted = (all_outputs > 0.5).astype(float)

        recall = recall_score(all_true_labels, all_predicted)
        f1 = f1_score(all_true_labels, all_predicted)
        precision = precision_score(all_true_labels, all_predicted)

        try:
            auc_score = roc_auc_score(all_true_labels, all_outputs)
        except ValueError:
            auc_score = 0

        try:
            precision_curve, recall_curve, _ = precision_recall_curve(all_true_labels, all_outputs)
            aupr = auc(recall_curve, precision_curve)
        except ValueError:
            aupr = 0

        accuracy = correct_predictions / total_predictions

    def aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    ACC_ci = bootstrap_ci(all_true_labels, all_predicted, accuracy_score)
    # print(f"Accuracy 95% CI: {ACC_ci}")

    AUC_ci = bootstrap_ci(all_true_labels, all_outputs, roc_auc_score)
    # print(f"AUC 95% CI: {AUC_ci}")

    AUPR_ci = bootstrap_ci(all_true_labels, all_outputs, aupr_score)
    # print(f"AUPR_ 95% CI: {AUPR_ci}")

    F1_ci = bootstrap_ci(all_true_labels, all_predicted, f1_score)
    # print(f"F1 95% CI: {F1_ci}")

    precision_ci = bootstrap_ci(all_true_labels, all_predicted, precision_score)
    # print(f"precision 95% CI: {precision_ci}")

    recall_ci = bootstrap_ci(all_true_labels, all_predicted, recall_score)
    # print(f"recall 95% CI: {recall_ci}")

    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        print(f'{epoch + 1},b {best_val_accuracy}\n')
        torch.save(model.state_dict(), f'../best_model/{args.dataset}/best_model_BEST.pth')
    with open(log_save_path, 'a') as f:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if args.dataset == 'Drugbank':
            f.write(f'L:2 B:{args.batch_size}  p:{args.p}  lr:{args.lr}  \n')
        else:
            f.write(
                f'L:3 B:{args.batch_size}  p:{args.p}  lr:{args.lr}  \n'
            )
        f.write(
            f'Time {current_time},Epoch [{epoch + 1}/{num_epochs}] Accuracy of the model on the valid set: {accuracy * 100:.2f}%\n')
        f.write(
            f'Epoch [{epoch + 1}/{num_epochs}]ACC: {accuracy:.4f} AUC: {auc_score:.4f} AUPR: {aupr:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1:.4f}\n')
        f.write(
            f'Epoch [{epoch + 1}/{num_epochs}]Accuracy 95% CI: {ACC_ci} AUC 95% CI: {AUC_ci} AUPR 95% CI: {AUPR_ci} precision 95% CI: {precision_ci} recall 95% CI: {recall_ci} F1 95% CI: {F1_ci}\n')
        f.write("\n")
    return best_val_accuracy


def test(dataset):
    log_save_path = ''
    if dataset == 'Drugbank':
        log_save_path = '../../Log/Drugbank/HY_drugbank_test_log.txt'
    elif dataset == 'KEGG_DRUG':
        log_save_path = '../../Log/KEGG_DRUG/HY_kegg_drug_test_log.txt'

    correct_predictions = 0
    total_predictions = 0
    all_true_labels = []
    all_outputs = []

    model.load_state_dict(torch.load(f'../best_model/{dataset}/best_model_BEST.pth'))

    with torch.no_grad():
        for node1_features, node2_features, drug1_text_emb, drug2_text_emb, labels in test_data_loader:
            node1_features = node1_features.to(device).to(torch.float32)
            node2_features = node2_features.to(device).to(torch.float32)
            drug1_text_emb = drug1_text_emb.to(device).to(torch.float32)
            drug2_text_emb = drug2_text_emb.to(device).to(torch.float32)
            labels = labels.to(device)

            outputs = model(node1_features, node2_features, drug1_text_emb, drug2_text_emb)
            predicted = (outputs.data > 0.5).float()

            total_predictions += labels.size(0)
            correct_predictions += (predicted.squeeze(1) == labels).sum().item()

            all_true_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.squeeze(1).cpu().numpy())

    all_true_labels = np.array(all_true_labels)
    all_outputs = np.array(all_outputs)
    all_predicted = (all_outputs > 0.5).astype(float)

    recall = recall_score(all_true_labels, all_predicted)
    f1 = f1_score(all_true_labels, all_predicted)
    precision = precision_score(all_true_labels, all_predicted)

    try:
        auc_score = roc_auc_score(all_true_labels, all_outputs)
    except ValueError:
        auc_score = 0

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(all_true_labels, all_outputs)
        aupr = auc(recall_curve, precision_curve)
    except ValueError:
        aupr = 0

    accuracy = correct_predictions / total_predictions

    def aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    ACC_ci = bootstrap_ci(all_true_labels, all_predicted, accuracy_score)
    # print(f"Accuracy 95% CI: {ACC_ci}")

    AUC_ci = bootstrap_ci(all_true_labels, all_outputs, roc_auc_score)
    # print(f"AUC 95% CI: {AUC_ci}")

    AUPR_ci = bootstrap_ci(all_true_labels, all_outputs, aupr_score)
    # print(f"AUPR_ 95% CI: {AUPR_ci}")

    F1_ci = bootstrap_ci(all_true_labels, all_predicted, f1_score)
    # print(f"F1 95% CI: {F1_ci}")

    precision_ci = bootstrap_ci(all_true_labels, all_predicted, precision_score)
    # print(f"precision 95% CI: {precision_ci}")

    recall_ci = bootstrap_ci(all_true_labels, all_predicted, recall_score)
    # print(f"recall 95% CI: {recall_ci}")
    with open(log_save_path, 'a') as f:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if args.dataset == 'Drugbank':
            f.write(f'L:2 B:{args.batch_size}  p:{args.p}  lr:{args.lr}  \n')
        else:
            f.write(
                f'L:3 B:{args.batch_size}  p:{args.p}  lr:{args.lr}  \n'
            )
        f.write(f'Time {current_time}, Accuracy of the model on the test set: {accuracy * 100:.2f}%\n')
        f.write(f'ACC: {accuracy:.4f}  AUC: {auc_score:.4f}  AUPR: {aupr:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1 Score: {f1:.4f}\n')
        f.write(
            f'Accuracy 95% CI: {ACC_ci} AUC 95% CI: {AUC_ci} AUPR 95% CI: {AUPR_ci} precision 95% CI: {precision_ci} recall 95% CI: {recall_ci} F1 95% CI: {F1_ci}\n')
        f.write("\n")


def main():
    best_val_accuracy = 0.0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train(epoch=epoch, num_epochs=num_epochs, dataset=args.dataset)
        # Epoch.append(epoch+1)
        model.eval()
        best_val_accuracy = valid(epoch=epoch, num_epochs=num_epochs, dataset=args.dataset,
                                  best_val_accuracy=best_val_accuracy)
    model.eval()
    test(dataset=args.dataset)


main()
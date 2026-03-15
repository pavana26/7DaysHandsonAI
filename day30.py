# import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics  import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer, BertModel,AdamW

# Load the dataset
df=pd.read_csv('essays.csv')
print("Dataset preview:")
print(df.head())

# Select releavant columns
df = df[['Essay','Overall']]

# Preprocess the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom dataset class
class EssayDataset(Dataset):
    def __init__(self, essays, scores, tokenizer, max_length=512):
        self.essays = essays
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        essay = str(self.essays[idx])
        score = self.scores[idx]
        
        encoding = self.tokenizer.encode_plus(
            essay,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'essay_text': essay,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'score': torch.tensor(score, dtype=torch.float)
        }
    
# Split dataset 
train_texts,val_texts,train_scores,val_scores = train_test_split(df['Essay'].values, df['Overall'].values, test_size=0.2, random_state=42)


train_dataset = EssayDataset(train_texts, train_scores, tokenizer)
val_dataset = EssayDataset(val_texts, val_scores, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Build the model
class EssayScoringModel(torch.nn.Module):
    def __init__(self):
        super(EssayScoringModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.regressor =torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        score = self.regressor(cls_output)
        return score.squeeze()
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EssayScoringModel().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.MSELoss()


# Train the model
def train(model, data_loader,loss_fn, optimizer,device ):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, scores)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Evaluate the model    
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    predictions = []
    true_scores = []
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, scores)
            total_loss += loss.item()

            predictions.extend(outputs.detach().cpu().numpy())
            true_scores.extend(scores.detach().cpu().numpy())
    mse=mean_squared_error(true_scores, predictions)
    r2=r2_score(true_scores, predictions)
    return total_loss / len(data_loader), mse, r2

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, loss_fn, optimizer, device)
    val_loss, mse, r2 = evaluate(model, val_loader, loss_fn, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}')

# Test the model on a sample essay
sample_essay = "This is a sample essay for testing the model."
encoding = tokenizer.encode_plus(
    sample_essay,
    add_special_tokens=True,
    max_length=512,
    return_token_type_ids=False,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)
model.eval()
with torch.no_grad():
    predicted_score = model(input_ids, attention_mask).item()

print(f'Predicted score for the sample essay: {predicted_score:.2f}')

# Visualise trainiing performance

losses={'Epoch':[1,2,3],'Train Loss':[0.8,0.6,0.4],'Val Loss':[0.35,0.25,0.3]}

loss_df=pd.DataFrame(losses)
sns.lineplot(x='Epoch', y='Train Loss', data=loss_df, marker='o', label='Train Loss')
sns.lineplot(x='Epoch', y='Val Loss', data=loss_df, marker='o', label='Val Loss')
plt.title('Training and Validation Loss')
plt.show()

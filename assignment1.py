import os
import json
import numpy as np
import miditoolkit
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import librosa
from tqdm import tqdm

# Constants
DATAROOT1 = "student_files/task1_composer_classification/"
DATAROOT2 = "student_files/task2_next_sequence_prediction/"
DATAROOT3 = "student_files/task3_audio_classification/"

# Audio processing constants
SAMPLE_RATE = 16000
N_MELS = 128  # Increased from baseline's 64
AUDIO_DURATION = 10  # seconds
BATCH_SIZE = 32

# Tags for Task 3
TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance', 'blues', 'punk', 'chill', 'electronic', 'country']

def extract_features(path):
    midi_obj = miditoolkit.midi.parser.MidiFile(DATAROOT1 + '/' + path)
    notes = midi_obj.instruments[0].notes
    num_notes = len(notes)
    if num_notes == 0:
        return [0] * 20  # Increased feature count
    
    # Basic features
    pitches = [note.pitch for note in notes]
    durations = [note.end - note.start for note in notes]
    average_pitch = np.mean(pitches)
    average_duration = np.mean(durations)
    note_density = num_notes / (midi_obj.max_tick / midi_obj.ticks_per_beat)
    pitch_range = max(pitches) - min(pitches)
    instrument_count = len(midi_obj.instruments)
    
    # Rhythm features
    time_diffs = np.diff([note.start for note in notes])
    rhythm_std = np.std(time_diffs) if len(time_diffs) > 0 else 0
    rhythm_mean = np.mean(time_diffs) if len(time_diffs) > 0 else 0
    rhythm_entropy = -np.sum(np.histogram(time_diffs, bins=20, density=True)[0] * 
                           np.log2(np.histogram(time_diffs, bins=20, density=True)[0] + 1e-10))
    
    # Pitch features
    pitch_std = np.std(pitches)
    pitch_hist = Counter(pitches)
    most_common_pitch = pitch_hist.most_common(1)[0][0] if pitch_hist else 0
    pitch_variety = len(pitch_hist)
    pitch_entropy = -np.sum(np.histogram(pitches, bins=20, density=True)[0] * 
                          np.log2(np.histogram(pitches, bins=20, density=True)[0] + 1e-10))
    
    # Duration features
    duration_std = np.std(durations)
    duration_hist = Counter(durations)
    most_common_duration = duration_hist.most_common(1)[0][0] if duration_hist else 0
    duration_variety = len(duration_hist)
    duration_entropy = -np.sum(np.histogram(durations, bins=20, density=True)[0] * 
                             np.log2(np.histogram(durations, bins=20, density=True)[0] + 1e-10))
    
    # Additional features
    total_duration = midi_obj.max_tick / midi_obj.ticks_per_beat
    avg_velocity = np.mean([note.velocity for note in notes])
    velocity_std = np.std([note.velocity for note in notes])
    
    # Chord features (simplified)
    chord_changes = 0
    for i in range(len(notes)-1):
        if abs(notes[i+1].pitch - notes[i].pitch) > 2:  # Simple chord change detection
            chord_changes += 1
    chord_change_rate = chord_changes / num_notes if num_notes > 0 else 0
    
    return [
        average_pitch,
        average_duration,
        note_density,
        pitch_range,
        instrument_count,
        rhythm_std,
        rhythm_mean,
        rhythm_entropy,
        pitch_std,
        most_common_pitch,
        pitch_variety,
        pitch_entropy,
        duration_std,
        most_common_duration,
        duration_variety,
        duration_entropy,
        total_duration,
        avg_velocity,
        velocity_std,
        chord_change_rate
    ]

def train_model(train_json_path):
    with open(train_json_path, 'r') as f:
        train_data = eval(f.read())
    
    # Extract features and labels
    X = [extract_features(k) for k in train_data]
    y = [train_data[k] for k in train_data]
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Task 1 validation accuracy: {val_acc:.4f}")
    
    # Check for overfitting
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"Task 1 training accuracy: {train_acc:.4f}")
    print(f"Task 1 train-val accuracy difference: {train_acc - val_acc:.4f}")
    
    # Retrain on full dataset
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X, y)
    return model

def predict(model, test_json_path, outpath=None):
    with open(test_json_path, 'r') as f:
        test_data = eval(f.read())
    predictions = {}
    for k in test_data:
        features = extract_features(k)
        pred = model.predict([features])[0]
        predictions[k] = str(pred)
    if outpath:
        with open(outpath, "w") as z:
            z.write(str(predictions) + '\n')
    return predictions

def accuracy(groundtruth, predictions):
    correct = 0
    for k in groundtruth:
        if k not in predictions:
            print("Missing " + str(k) + " from predictions")
            return 0
        if predictions[k] == groundtruth[k]:
            correct += 1
    return correct / len(groundtruth)

# Task 2 Implementation
def extract_features_task2(path):
    midi_obj = miditoolkit.midi.parser.MidiFile(DATAROOT2 + '/' + path)
    notes = midi_obj.instruments[0].notes
    if len(notes) == 0:
        return [0] * 10  # Default features if no notes
    
    # Basic features
    pitches = [note.pitch for note in notes]
    durations = [note.end - note.start for note in notes]
    
    # Pitch features
    pitch_mean = np.mean(pitches)
    pitch_std = np.std(pitches)
    pitch_range = max(pitches) - min(pitches)
    pitch_hist = Counter(pitches)
    pitch_variety = len(pitch_hist)
    
    # Duration features
    duration_mean = np.mean(durations)
    duration_std = np.std(durations)
    
    # Rhythm features
    time_diffs = np.diff([note.start for note in notes])
    rhythm_std = np.std(time_diffs) if len(time_diffs) > 0 else 0
    rhythm_mean = np.mean(time_diffs) if len(time_diffs) > 0 else 0
    
    return [
        pitch_mean,
        pitch_std,
        pitch_range,
        pitch_variety,
        duration_mean,
        duration_std,
        rhythm_std,
        rhythm_mean,
        len(notes),  # Number of notes
        midi_obj.max_tick / midi_obj.ticks_per_beat  # Duration in beats
    ]

def train_model_task2(train_json_path):
    with open(train_json_path, 'r') as f:
        train_data = eval(f.read())
    
    # Extract features and labels
    X = []
    y = []
    for (path1, path2), is_neighbor in train_data.items():
        features1 = extract_features_task2(path1)
        features2 = extract_features_task2(path2)
        combined_features = features1 + features2
        X.append(combined_features)
        y.append(is_neighbor)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Task 2 validation accuracy: {val_acc:.4f}")
    
    # Check for overfitting
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"Task 2 training accuracy: {train_acc:.4f}")
    print(f"Task 2 train-val accuracy difference: {train_acc - val_acc:.4f}")
    
    # Retrain on full dataset
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X, y)
    return model

def predict_task2(model, test_json_path, outpath=None):
    with open(test_json_path, 'r') as f:
        test_data = eval(f.read())
    
    predictions = {}
    for (path1, path2) in test_data:
        features1 = extract_features_task2(path1)
        features2 = extract_features_task2(path2)
        combined_features = features1 + features2
        pred = bool(model.predict([combined_features])[0])  # Convert numpy bool to Python bool
        predictions[(path1, path2)] = pred
    
    if outpath:
        with open(outpath, "w") as z:
            z.write(str(predictions) + '\n')
    return predictions

def accuracy_task2(groundtruth, predictions):
    correct = 0
    for k in groundtruth:
        if k not in predictions:
            print("Missing " + str(k) + " from predictions")
            return 0
        if predictions[k] == groundtruth[k]:
            correct += 1
    return correct / len(groundtruth)

# Task 3 Implementation
class AudioDataset(Dataset):
    def __init__(self, meta, preload=True):
        self.meta = meta
        self.ks = list(meta.keys())
        self.idToPath = dict(zip(range(len(self.ks)), self.ks))
        self.pathToFeat = {}
        
        self.mel = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=2048,
            hop_length=512
        )
        self.db = AmplitudeToDB()
        
        self.preload = preload
        if self.preload:
            for path in tqdm(self.ks, desc="Preloading features"):
                waveform = self.extract_waveform(path)
                mel_spec = self.db(self.mel(waveform)).squeeze(0)
                self.pathToFeat[path] = mel_spec

    def extract_waveform(self, path):
        waveform, sr = librosa.load(DATAROOT3 + '/' + path, sr=SAMPLE_RATE)
        waveform = torch.FloatTensor(waveform).unsqueeze(0)
        
        # Pad or trim to target length
        target_len = SAMPLE_RATE * AUDIO_DURATION
        if waveform.shape[1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_len]
        return waveform

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        path = self.idToPath[idx]
        tags = self.meta[path]
        bin_label = torch.tensor([1 if tag in tags else 0 for tag in TAGS], dtype=torch.float32)
        
        if self.preload:
            mel_spec = self.pathToFeat[path]
        else:
            waveform = self.extract_waveform(path)
            mel_spec = self.db(self.mel(waveform)).squeeze(0)
        
        return mel_spec.unsqueeze(0), bin_label, path

class ImprovedCNN(nn.Module):
    def __init__(self, n_classes=len(TAGS)):
        super(ImprovedCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size of flattened features
        self._to_linear = None
        self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, n_classes)

    def _get_conv_output_size(self):
        # Create a dummy input to calculate the size
        x = torch.randn(1, 1, N_MELS, AUDIO_DURATION * SAMPLE_RATE // 512 + 1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return torch.sigmoid(x)  # Sigmoid for multilabel classification

class AudioPipeline:
    def __init__(self, model, learning_rate=1e-4, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        self.criterion = nn.BCELoss()
        
    def train(self, train_loader, val_loader, num_epochs=30):
        best_val_map = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            
            for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            # Evaluate on validation set
            val_predictions, val_map = self.evaluate(val_loader)
            print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f}, Val mAP: {val_map:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_map)
            
            # Save best model
            if val_map > best_val_map:
                best_val_map = val_map
                torch.save(self.model.state_dict(), "best_model_task3.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    def evaluate(self, loader, threshold=0.5, outpath=None):
        self.model.eval()
        preds, targets, paths = [], [], []
        
        with torch.no_grad():
            for x, y, ps in loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                preds.append(outputs.cpu())
                targets.append(y.cpu())
                paths.extend(ps)
        
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        
        # Convert predictions to binary using threshold
        preds_bin = (preds > threshold).float()
        
        # Create predictions dictionary
        predictions = {}
        for i in range(preds_bin.shape[0]):
            predictions[paths[i]] = [TAGS[j] for j in range(len(preds_bin[i])) if preds_bin[i][j]]
        
        # Calculate mAP if not saving predictions
        mAP = None
        if not outpath:
            mAP = average_precision_score(targets, preds, average='macro')
        else:
            with open(outpath, "w") as z:
                z.write(str(predictions) + '\n')
        
        return predictions, mAP

def run_task3():
    # Load data
    with open(DATAROOT3 + "/train.json", 'r') as f:
        train_data = eval(f.read())
    with open(DATAROOT3 + "/test.json", 'r') as f:
        test_data = eval(f.read())
    
    # Create datasets
    full_train = AudioDataset(train_data)
    # Convert test data list to dictionary format
    test_data_dict = {path: [] for path in test_data}
    test_set = AudioDataset(test_data_dict)
    
    # Split training data into train and validation
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    
    # Initialize model and pipeline
    model = ImprovedCNN()
    pipeline = AudioPipeline(model, learning_rate=1e-4)
    
    # Train model
    pipeline.train(train_loader, val_loader, num_epochs=30)
    
    # Load best model
    model.load_state_dict(torch.load("best_model_task3.pth"))
    
    # Evaluate on test set
    test_preds, _ = pipeline.evaluate(test_loader, outpath="predictions3.json")
    
    # Calculate training mAP
    train_preds, train_map = pipeline.evaluate(train_loader)
    print(f"Task 3 training mAP = {train_map:.4f}")

if __name__ == "__main__":
    # Task 1
    print("Running Task 1...")
    model1 = train_model(DATAROOT1 + "/train.json")
    train_preds1 = predict(model1, DATAROOT1 + "/train.json")
    test_preds1 = predict(model1, DATAROOT1 + "/test.json", "predictions1.json")
    with open(DATAROOT1 + "/train.json", 'r') as f:
        train_labels1 = eval(f.read())
    acc1 = accuracy(train_labels1, train_preds1)
    print("Task 1 training accuracy = " + str(acc1))
    
    # Task 2
    print("\nRunning Task 2...")
    model2 = train_model_task2(DATAROOT2 + "/train.json")
    train_preds2 = predict_task2(model2, DATAROOT2 + "/train.json")
    test_preds2 = predict_task2(model2, DATAROOT2 + "/test.json", "predictions2.json")
    with open(DATAROOT2 + "/train.json", 'r') as f:
        train_labels2 = eval(f.read())
    acc2 = accuracy_task2(train_labels2, train_preds2)
    print("Task 2 training accuracy = " + str(acc2))
    
    # # Task 3
    # print("\nRunning Task 3...")
    # run_task3() 
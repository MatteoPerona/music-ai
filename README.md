# Music AI Analysis

A collection of machine learning models for music analysis and classification tasks. This project implements three distinct music analysis tasks using different approaches to feature extraction and model architecture.

## Tasks

### 1. Composer Classification
- Analyzes MIDI files to identify the composer
- Uses a sophisticated feature extraction system that captures:
  - Rhythm patterns and entropy
  - Pitch statistics and distributions
  - Duration patterns
  - Chord progression characteristics
  - Velocity dynamics
- Implements a Random Forest classifier with 200 trees

### 2. Sequence Coherence Detection
- Determines if two musical sequences are neighbors in a real piece
- Extracts features from both sequences including:
  - Pitch statistics
  - Duration patterns
  - Rhythm features
- Uses a binary classifier to learn the relationship between sequences

### 3. Music Genre Tagging
- Multi-label classification of audio files into genres
- Features:
  - Mel spectrogram representation
  - Deep CNN architecture with batch normalization
  - Multi-label classification with sigmoid outputs
- Implements early stopping and learning rate scheduling

## Technical Details

### Dependencies
- PyTorch
- scikit-learn
- librosa
- miditoolkit
- numpy
- torchaudio

### Model Architecture
- Task 1 & 2: Random Forest with 200 trees
- Task 3: CNN with 4 convolutional layers and batch normalization

### Feature Extraction
- MIDI features for symbolic music analysis
- Mel spectrograms for audio analysis
- Custom feature engineering for musical patterns

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the models:
```bash
python assignment1.py
```

## Results
The models are evaluated using:
- Accuracy for composer classification
- Binary accuracy for sequence coherence
- Mean Average Precision (mAP) for genre tagging

## TODO
- [ ] Add link to dataset and data preprocessing instructions

## License
MIT License 
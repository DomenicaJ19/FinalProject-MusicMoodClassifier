# Music Mood Classifier
Created by: Domenica Jaramillo and Monique Villadiego

This project is  machine learning project that classifies music into different mood categories using audio features from a CSV dataset and a Random Forest classifier.

Time spent: 7 hours spent in total

## Project Overview

This project uses Python and Google Colab to classify music into five mood categories:
- **Happy** - Upbeat, cheerful music
- **Sad** - Melancholic, emotional music
- **Energetic** - High-energy, fast-paced music
- **Calm** - Relaxing, peaceful music
- **Angry** - Aggressive, intense music

## Features

- **Audio Features**: Uses pre-extracted audio features from the CSV dataset:
  - Danceability - How suitable for dancing (0-1)
  - Energy - Perceptual intensity (0-1)
  - Valence - Musical positiveness (0-1)
  - Tempo - Overall estimated tempo (BPM)
  - Loudness - Overall loudness in decibels
  - Speechiness - Presence of spoken words (0-1)
  - Acousticness - Acoustic confidence (0-1)
  - Instrumentalness - Predicts if track has vocals (0-1)
  - Liveness - Detects audience presence (0-1)
  - Key - Key the track is in (0-11)
  - Time signature - Estimated time signature
  - Duration - Song length in milliseconds

- **Machine Learning Model**: Implements Random Forest Classifier with:
  - 100 estimators
  - Feature scaling using StandardScaler
  - Train/test split with stratification
  - Feature importance analysis

- **Evaluation Tools**: Includes accuracy metrics, classification reports, and example predictions

## Project Walkthrough Presentation

- https://drive.google.com/file/d/16VGrcNluBB4Kh4H8vHsoN0EXsrs1yovB/view?usp=sharing

## Project Walkthrough Demo

- https://drive.google.com/file/d/1bHZUxA57QV4YQ3jj8zGprcDYuW129sl7/view?usp=sharing

## Getting Started

### Prerequisites

- Google Colab account (free)
- `data_moods.csv` dataset file
- Basic knowledge of Python and machine learning

### Installation

1. **Open the Notebook in Google Colab**:
   - Upload `MusicMoodClassifier.ipynb` to Google Colab
   - Or open it directly from Google Drive

2. **Install Required Libraries**:
   The notebook will automatically install all required libraries when you run the first cell:
   ```python
   !pip install pandas numpy scikit-learn -q
   ```

3. **Upload the Dataset**:
   - Upload `data_moods.csv` to your Colab environment
   - The dataset contains 686 songs with pre-extracted audio features

### Usage

1. **Load the Dataset**:
   - The notebook loads the `data_moods.csv` file
   - The dataset includes songs labeled with their mood categories

2. **Run the Notebook**:
   - Execute cells sequentially from top to bottom
   - The notebook will:
     - Load and explore the dataset
     - Prepare features for machine learning
     - Train a Random Forest classifier
     - Evaluate model performance
     - Show example predictions

3. **Train the Model**:
   - The Random Forest classifier trains automatically
   - Features are scaled using StandardScaler
   - Data is split into 80% training and 20% testing sets

4. **Make Predictions**:
   - Use the `predict_mood_from_features()` function to classify new songs
   - Provide a dictionary with audio features (danceability, energy, valence, etc.)
   - Get the predicted mood and confidence scores for each mood category

## Project Structure

```
Music_Mood_Classifier/
├── MusicMoodClassifier.ipynb     # Main Colab notebook
├── data_moods.csv                # Dataset with 686 songs and features
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## Dataset

### Current Dataset

The project uses `data_moods.csv` which contains:
- **686 songs** with pre-extracted audio features
- **5 mood categories**: Happy, Sad, Energetic, Calm, Angry
- **12 audio features** per song (danceability, energy, valence, tempo, etc.)
- **Song metadata**: name, artist, album, release date, popularity

### Dataset Columns

- `name` - Song name
- `album` - Album name
- `artist` - Artist name
- `id` - Track ID
- `release_date` - Release date
- `popularity` - Popularity score
- `length` - Song length in milliseconds
- `danceability`, `energy`, `valence`, `tempo`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `key`, `time_signature` - Audio features
- `mood` - **Label** (Happy, Sad, Energetic, Calm, Angry)

### Creating Your Own Dataset

To create a custom dataset:
1. Extract audio features for your songs (using any audio analysis tool or library)
2. Manually label each song with a mood category
3. Save as CSV with the same column structure
4. Ensure balanced representation across mood categories

## Tips for Better Results

1. **More Data**: The more training samples per mood, the better your model will perform
2. **Balanced Dataset**: Try to have similar numbers of samples for each mood category
3. **Feature Selection**: Experiment with different combinations of audio features
4. **Hyperparameter Tuning**: Adjust Random Forest parameters (n_estimators, max_depth) for better performance
5. **Feature Scaling**: The model uses StandardScaler - ensure all features are properly scaled
6. **Cross-Validation**: Consider using k-fold cross-validation for more robust evaluation

## Model Performance

The performance of the Random Forest model depends on:
- Size and quality of your dataset
- Balance of classes across mood categories
- Feature selection and quality
- Model hyperparameters (n_estimators, max_depth)

With the current dataset (686 songs):
- The model uses 100 estimators with max_depth=20
- Features are standardized before training
- 80/20 train/test split with stratification
- Check the classification report for per-class performance metrics

## Example Usage

After training, predict mood for a new song:

```python
# Example features for a new song
new_song_features = {
    'danceability': 0.8,
    'energy': 0.7,
    'key': 5,
    'loudness': -5.0,
    'speechiness': 0.05,
    'acousticness': 0.2,
    'instrumentalness': 0.0,
    'liveness': 0.1,
    'valence': 0.8,  # High valence = happy
    'tempo': 120,
    'time_signature': 4,
    'duration_ms': 200000
}

# Predict mood
mood, confidence = predict_mood_from_features(new_song_features)
print(f"Predicted Mood: {mood}")
print(f"Confidence: {confidence[mood]:.2%}")
```

## Troubleshooting

### Common Issues

1. **CSV File Not Found**:
   - Ensure `data_moods.csv` is uploaded to your Colab environment
   - Check the file path matches the code: `pd.read_csv('data_moods.csv')`
   - Verify the file is in the same directory as the notebook

2. **Missing Values**:
   - The notebook automatically removes rows with missing values
   - Check the dataset for any corrupted entries
   - Verify all required feature columns are present

3. **Low Accuracy**:
   - Increase dataset size (more songs per mood category)
   - Balance your classes (similar number of songs per mood)
   - Try different feature combinations
   - Tune hyperparameters (n_estimators, max_depth)
   - Check feature importance to identify most relevant features

4. **Import Errors**:
   - Install required packages: `!pip install pandas numpy scikit-learn`
   - Restart runtime if packages were just installed

## Future Improvements

- Add more mood categories
- Implement additional ML models (SVM, Neural Networks) for comparison
- Create a web interface for predictions
- Implement model persistence (save/load trained models)
- Add data augmentation techniques
- Implement cross-validation for more robust evaluation

## License

Copyright [2025] [Domenica Jaramillo, Monique Villadiego]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
---

**Good luck with your project! 🎵**

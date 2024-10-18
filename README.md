# NLP Arabic Dialects Classification Project

This project focuses on classifying Arabic dialects, including Egyptian (EG), Lebanese (LB), Libyan (LY), Moroccan (MA), and Sudanese (SD). We aim to develop models that can accurately distinguish between these dialects based on text data.

| Dialect Code | Dialect Name |
|--------------|--------------|
| 0            | EG (Egyptian)|
| 1            | LB (Lebanese)|
| 2            | LY (Libyan)  |
| 3            | MA (Moroccan)|
| 4            | SD (Sudanese)|

## Video Demonstration

![Video Demonstration](https://github.com/Nadaalswah/NLP-Arabic-Dialects-Project/blob/main/Project%20GIF.gif)

## Dataset
We used the following dataset for this project:
- **Database Path**: `/kaggle/input/dialects-db/dialects_database.db`

### Objective
At this stage of the project, our goal is to retrieve and merge data from the SQLite database, creating a single cohesive dataframe that we can then save as a CSV file. This consolidated dataframe will simplify the subsequent data preprocessing and model training tasks.

## Project Structure

### 1. **Data Extraction (Notebook 1)**
In the first notebook, we extract the necessary data from the database to prepare for dialect identification. The key steps include:
- Connecting to the SQLite database.
- Retrieving data from the `id_text` and `id_dialect` tables.
- Merging the data into a single dataframe based on a common identifier.
- Saving the merged dataframe as a CSV file for further use.

### 2. **Data Preprocessing (Notebook 2)**
In this notebook, we apply various cleaning and preprocessing steps to prepare the data for modeling. The steps include:
- Removing Tatweel (elongation characters).
- Removing digits, symbols, emojis, and URLs.
- Removing usernames, non-Arabic characters, extra spaces, and hashtags.
- Handling un-ASCII characters.
- Addressing noise and garbage characters.
- Handling bidirectional text issues.

### 3. **Model Training (Notebook 3)**
We train both machine learning and deep learning models for dialect classification. In this notebook:
- We experimented with and without handling imbalanced data.
- **Logistic Regression**: We created a pipeline using `CountVectorizer` and `LogisticRegression` for text classification. We also used Grid Search for hyperparameter tuning.
- **LSTM Model**: We developed an LSTM model for dialect classification.
- Both models were saved for deployment.

### Advanced Models
We also explored other models before finalizing the best two, which are stored in the folder `advanced Notebook`.

## How to Run the Project

1. **Clone the repository**: Download or clone this repo to your local machine.
2. **Install the requirements**: Run the following command to install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   python app.py

## Team Members
- Nada Alswah
- Ahmed El-Metwally
- Amgad Shalaby
- Nour Raafat


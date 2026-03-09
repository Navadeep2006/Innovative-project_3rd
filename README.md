# Food Calorie Predictor 🥗

A beautiful and intuitive Streamlit web application that predicts the calories of food items based on their macronutrient breakdown (Protein, Carbohydrate, Fiber, Sugar, and Fat) using a trained Random Forest Regressor model.

## Features

- **Search by Food Name**: Quickly lookup existing foods from the USDA Nutrient database (`food.csv`). View the nutritional breakdown, actual calories vs predicted calories, and accuracy.
- **Manual Nutrient Entry**: Enter your own custom nutrient values per 100g to instantly estimate total calories.
- **Visual Insights**: Provides clear UI cards, badges, and percentage progress bars showing the calorie contribution of each macronutrient.

## Files
- `app.py`: The main Streamlit web application.
- `food.csv`: The dataset containing food items and their nutritional features.
- `calorie_model.pkl`: A pre-trained Random Forest Regressor model.
- `foodcalarie.ipynb`: The Jupyter Notebook used for data exploration and model training.

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Navadeep2006/Innovative-project_3rd.git
   cd Innovative-project_3rd
   ```

2. **Install the dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application locally:
```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.

## Notes
- To ensure the app functions completely, verify that `food.csv` (used for the search feature) is located in the same directory.
- The pre-trained model (`calorie_model.pkl`) acts as a fallback when `food.csv` is absent, skipping fresh training to load instantly.

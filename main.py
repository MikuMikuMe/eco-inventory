Creating a complete Python program for an eco-inventory system that uses predictive analytics to optimize stock levels is a complex task that typically requires integration with data sources, a prediction model, and a user interface or API. Here, I'll provide a basic example with a simple predictive model using linear regression from scikit-learn, and a console-based interface. This will serve as a simplified baseline you can expand upon.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging

# Setup logging
logging.basicConfig(filename='eco_inventory_log.txt', level=logging.DEBUG, 
                    format='%(levelname)s:%(message)s')


class EcoInventory:
    def __init__(self, data_file):
        """Initializes the EcoInventory system."""
        self.data_file = data_file
        self.model = None
        self.data = self._load_data()

    def _load_data(self):
        """Loads data from a CSV file."""
        try:
            data = pd.read_csv(self.data_file)
            logging.info('Data loaded successfully from %s', self.data_file)
            return data
        except Exception as e:
            logging.error('Error loading data: %s', e)
            raise

    def prepare_data(self):
        """Prepares the data for the model."""
        try:
            # Assuming data has 'historical_demand' and 'current_stock_level' columns
            X = self.data['historical_demand'].values.reshape(-1, 1)
            y = self.data['current_stock_level'].values

            return train_test_split(X, y, test_size=0.2, random_state=42)
        except KeyError as e:
            logging.error('Missing required data column: %s', e)
            raise
        except Exception as e:
            logging.error('Error preparing data: %s', e)
            raise

    def train_model(self, X_train, y_train):
        """Trains a linear regression model."""
        try:
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            logging.info('Model trained successfully.')
        except Exception as e:
            logging.error('Error training model: %s', e)
            raise

    def predict_stock_level(self, historical_demand):
        """Predicts stock level based on historical demand."""
        try:
            X_new = np.array(historical_demand).reshape(-1, 1)
            prediction = self.model.predict(X_new)
            logging.info('Prediction made for historical demand: %s', historical_demand)
            return prediction
        except Exception as e:
            logging.error('Error making prediction: %s', e)
            raise

    def evaluate_model(self, X_test, y_test):
        """Evaluates the model using Mean Squared Error."""
        try:
            predictions = self.model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            logging.info('Model evaluation completed with MSE: %f', mse)
            return mse
        except Exception as e:
            logging.error('Error evaluating model: %s', e)
            raise


if __name__ == "__main__":
    # Example usage:
    inventory_system = EcoInventory(data_file='inventory_data.csv')

    try:
        # Prepare the data
        X_train, X_test, y_train, y_test = inventory_system.prepare_data()

        # Train the model
        inventory_system.train_model(X_train, y_train)

        # Evaluate the model
        mse = inventory_system.evaluate_model(X_test, y_test)
        print(f'Model Mean Squared Error: {mse}')

        # Predict stock levels
        example_demand = [100, 150, 200]  # Example demands to predict stock for
        predictions = inventory_system.predict_stock_level(example_demand)
        print('Predicted Stock Levels:', predictions)

    except Exception as e:
        print(f'An error occurred: {e}')
        logging.error('Unexpected error: %s', e)
```

### Key Components:

1. **Data Handling**: This example loads data from a CSV file. Ensure your `inventory_data.csv` has the necessary columns for training, or modify the code accordingly.

2. **Error Handling**: Basic error handling with logging is included for file operations, data processing, model training, and prediction.

3. **Predictive Model**: A linear regression model is used to predict stock levels based on historical demand. You can expand this with more sophisticated models as needed.

4. **Logging**: Logs operations to a file for monitoring.

You can extend this basic structure by adding more advanced features like time-series forecasting, integrating with inventory databases, or developing a front-end application or REST API for user interaction.
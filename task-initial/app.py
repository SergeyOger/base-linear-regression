from __future__ import annotations
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import os

def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> ArrayLike:
    """
    Deserialize and load the regression model and use it to predict on user provided data.

    This function takes a file name 'filename' that has a default value.
    It uses Joblib 'load' to load the model using the provided file name.
    When the model is loaded, call its `predict` method on provied data.

    Args:
        X (array-like): User provided data used for prediction.
        filename (str): Name of the file that is used to store the model.

    Returns:
        np.ndarray: Predicted value.
    """

    filepath = os.path.join(os.path.dirname(__file__), filename)
    model = load(filepath)
    y = model.predict(X)

    return y

def create_streamlit_app():
    """
    Creates a Streamlit web application for making predictions with a simple regression model.

    This function sets up a Streamlit app with a user interface for inputting a single feature
    value and making predictions using a pre-trained regression model. The app includes:

    - A title displayed at the top of the app.
    - A slider for the user to select an input feature value within a specified range (-3.0 to 3.0).
    - A "Predict value" button that, when clicked, triggers the prediction process.
    - Upon clicking the "Predict value" button, the function:
        - Calls `load_and_predict`, passing the selected feature as input, to load the regression model
          and make a prediction.
        - Displays the prediction result on the app.
        - Calls `visualize_difference`, passing the input feature and the prediction result,
          to visualize the difference between the predicted value and the actual value in the original dataset.

    Note: This function does not return any value. It directly manipulates the Streamlit app's UI by
    writing content and rendering UI elements.
    """

    # Streamlit app title
    st.title("Linear Regression Model Prediction")

    # User input for new prediction using a slider
    input_feature = st.slider("Select feature value:", min_value=-3.0, max_value=3.0, step=0.1)

    # Button to make a prediction
    if st.button("Predict value"):
        prediction = load_and_predict([[input_feature]])

        # 2. Display the prediction.
        st.write(f"Prediction: {prediction[0]:.2f}")

        # 4. Call visualize_difference to display a plot visualizing the difference between actual and perdicted value.

        visualize_difference(input_feature, prediction[0])

def visualize_difference(input_feature: float, prediction: ArrayLike):
    """
    Deserialize and load the initial datasets. Calculate the difference between actual data
    in the 'y' dataset and the predicted value for a given 'input_feature'.

    Visualize the difference by plotting the entire 'X' & 'y' as a Scatter plot. Then add
    a blue dot that represents the actual target value, and a red dot that represents the predicted target value for the given 'input_feature'.
    Add a dashed line connects these points, highlighting the difference between them, which is annotated on the plot.

    Args:
        input_feature (float): User provided data used for prediction.
        prediction (array-like): Predicted value.

    """
    # Load the X and y datasets
    X_filename = "X.joblib"
    y_filename = "y.joblib"

    X = load(os.path.join(os.path.dirname(__file__), X_filename))

    y = load(os.path.join(os.path.dirname(__file__), y_filename))

    actual_target = y[_index_of_closest(X, input_feature)]

    # Calculate difference
    difference = actual_target - prediction

    # Visualization
    fig = plt.figure(figsize=(6, 4))

    plt.scatter(X, y, color='grey', label='Dataset', alpha=0.6)
    plt.scatter(input_feature, actual_target, color='blue', s=100, label='Actual target', zorder=5)
    plt.scatter(input_feature, prediction, color='red', s=100, label='Predicted target', zorder=5)
    plt.legend()
    plt.title('Linear Regression Model Prediction')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.grid(True, alpha=0.3)
    plt.plot([input_feature, input_feature], [actual_target, prediction], 'k--', linewidth=2)
    plt.annotate(f'Difference: {difference:.2f}',
                 xy=(input_feature, (actual_target + prediction) / 2),
                 xytext=(10, 0), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    st.pyplot(fig)

# This is a helper function. No need to edit it
def _index_of_closest(X: ArrayLike, k: float) -> int:
    """
    This function takes an array-like object `X` and a float `k`, and returns the index of the
    element in `X` that is closest to `k`. The function first converts `X` into a NumPy array
    (if it isn't one already) to ensure compatibility with NumPy operations. It then calculates
    the absolute difference between each element in `X` and `k`, identifies the minimum value
    among these differences, and returns the index of this minimum difference.

    Args:
        X (ArrayLike): An array-like object containing numerical data. It can be a list, tuple,
      or any object that can be converted to a NumPy array.
        k (float): The target value to which the closest element in `X` is sought.

    Returns:
        int: The index of the element in `X` that is closest to the value `k`.
    Returns:
        int: Index for the closest value to k in X.
    Finds the index of the element in `X` that is closest to the value `k`.

    """
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx


if __name__ == '__main__':
    create_streamlit_app()

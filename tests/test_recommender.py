import sys
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Mock streamlit and decorators to allow importing app.py under test
def identity_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        # Called directly: @st.cache_resource
        return args[0]
    # Called with arguments: @st.cache_resource(ttl=3600)
    def decorator(func):
        return func
    return decorator

# Dictionary-like mock for st.session_state to allow .get(), attribute access, and indexing
class SessionStateMock(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        return MagicMock()
    def __setattr__(self, key, value):
        self[key] = value

st_mock = MagicMock()
st_mock.cache_data = identity_decorator
st_mock.cache_resource = identity_decorator
st_mock.session_state = SessionStateMock()

# Setup widget mock returns to prevent type errors during module import & formatting
st_mock.slider.return_value = 0.6
st_mock.selectbox.return_value = "Arjun (User 1)"
st_mock.radio.return_value = "All"
st_mock.button.return_value = False
st_mock.text_input.return_value = ""

# Configure st.tabs to return 6 items to avoid module-level unpacking ValueError in app.py
st_mock.tabs.return_value = [MagicMock() for _ in range(6)]

# Dynamic columns mock to return the correct number of mock objects when unpacking
def mock_columns(spec):
    if isinstance(spec, int):
        return [MagicMock() for _ in range(spec)]
    elif isinstance(spec, (list, tuple)):
        return [MagicMock() for _ in range(len(spec))]
    return [MagicMock(), MagicMock()]

st_mock.columns.side_effect = mock_columns

sys.modules["streamlit"] = st_mock
sys.modules["streamlit.components.v1"] = MagicMock()

# Mock tensorflow's load_model to avoid loading the heavy .h5 file during test initialization
tf_models_mock = MagicMock()
tf_models_mock.load_model.return_value = MagicMock()
sys.modules["tensorflow.keras.models"] = tf_models_mock

# Now import the functions from app.py
from app import _clean_search_title, _initials, _card_gradient, cold_start_recs

def test_clean_search_title():
    assert _clean_search_title("Spirited Away (2001)") == "Spirited Away"
    assert _clean_search_title("Attack on Titan: Season 3 Part 2") == "Attack on Titan"
    assert _clean_search_title("My Hero Academia 4th Season") == "My Hero Academia"
    assert _clean_search_title("Normal Title") == "Normal Title"
    assert _clean_search_title("   ") == "   "

def test_initials():
    assert _initials("Spirited Away") == "SA"
    assert _initials("Cowboy Bebop") == "CB"
    assert _initials("Akira") == "A"
    assert _initials("The Lord of the Rings") == "TL"  # first 2 words: The Lord

def test_card_gradient():
    grad1, grad2 = _card_gradient("Spirited Away")
    assert grad1.startswith("rgba(")
    assert grad2.startswith("rgba(")
    
    # Hash-stability: check that same title always yields the same gradient colors
    grad1_again, grad2_again = _card_gradient("Spirited Away")
    assert grad1 == grad1_again
    assert grad2 == grad2_again

def test_cold_start_recs():
    # Setup dummy data for recommendation search
    df = pd.DataFrame([
        {"title": "Movie A", "type": "Movie", "description": "action adventure sci-fi"},
        {"title": "Movie B", "type": "Movie", "description": "comedy romance drama"},
        {"title": "Movie C", "type": "Movie", "description": "sci-fi spaceships futuristic"},
    ])
    
    # Mock NN model and TF-IDF matrix
    nn_mock = MagicMock()
    # Assume 3 neighbors, indices [2, 0, 1]
    nn_mock.kneighbors.return_value = (np.array([[0.1, 0.2, 0.3]]), np.array([[2, 0, 1]]))
    
    tfidf_matrix = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 2]])
    
    # Call cold_start_recs
    ratings = {"Movie A": 5}
    recs = cold_start_recs(ratings, df, nn_mock, tfidf_matrix, n=2)
    
    # Assertions
    assert isinstance(recs, pd.DataFrame)
    assert not recs.empty
    # The output should not contain "Movie A" since it was rated
    assert "Movie A" not in recs["title"].values
    assert "Movie C" in recs["title"].values or "Movie B" in recs["title"].values

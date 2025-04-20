import pytest
from unittest.mock import Mock


# -------------------------------
# FIXTURE: set up common input data
# -------------------------------
@pytest.fixture
def sample_numbers():
    return [1, 2, 3, 4, 5]


# -------------------------------
# FUNCTION TO TEST
# -------------------------------
def process_numbers(numbers, multiplier_func):
    """Applies a multiplier function to each number in the list."""
    return [multiplier_func(n) for n in numbers]


# -------------------------------
# TEST: using a stub (simple fake function)
# -------------------------------
def test_process_numbers_with_stub(sample_numbers):
    def stub_multiplier(x):
        return x * 10

    result = process_numbers(sample_numbers, stub_multiplier)

    assert result == [10, 20, 30, 40, 50]


# -------------------------------
# TEST: using a mock to inspect calls
# -------------------------------
def test_process_numbers_with_mock(sample_numbers):
    mock_func = Mock(side_effect=lambda x: x * 2)  # behaves like a multiplier

    result = process_numbers(sample_numbers, mock_func)

    assert result == [2, 4, 6, 8, 10]
    assert mock_func.call_count == len(sample_numbers)
    mock_func.assert_called_with(5)  # last item in list


# -------------------------------
# TEST: assertion fails if values are different
# -------------------------------
def test_assertion_failure_example():
    actual = 1 + 1
    expected = 3
    assert actual == expected, f"Expected {expected}, but got {actual}"

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.language_models.chat_models import BaseChatModel

from self_discover._iself_discover import (
    select,
    adapt,
    structured_planning,
    structured_application,
    unstructured_planning,
    unstructured_application,
    build_iself_discover_graph,
    iself_discover,
    ISelfDiscoverState,
)


@pytest.fixture
def mock_llm_model():
    return MagicMock(spec=BaseChatModel)


@pytest.fixture
def mock_state():
    return ISelfDiscoverState(
        task_description="Test task",
        answer_formats="Test format",
        selected_modules=None,
        adapted_modules=None,
        reasoning_plan=None,
        reasoning=None,
    )


@patch("self_discover._iself_discover.stream_response")
def test_select(mock_stream_response, mock_state):
    mock_stream_response.return_value = "selected_modules_result"
    result = select(mock_state)
    assert result == {"selected_modules": "selected_modules_result"}


@patch("self_discover._iself_discover.stream_response")
def test_adapt(mock_stream_response, mock_state):
    mock_stream_response.return_value = "adapted_modules_result"
    result = adapt(mock_state)
    assert result == {"adapted_modules": "adapted_modules_result"}


@patch("self_discover._iself_discover.stream_response")
def test_structured_planning(mock_stream_response, mock_state):
    mock_stream_response.return_value = "reasoning_structure_result"
    result = structured_planning(mock_state)
    assert result == {"reasoning_structure": "reasoning_structure_result"}


@patch("self_discover._iself_discover.stream_response")
def test_structured_application(mock_stream_response, mock_state):
    mock_stream_response.return_value = "reasoning_result"
    result = structured_application(mock_state)
    assert result == {"reasoning": "reasoning_result"}


@patch("self_discover._iself_discover.stream_response")
def test_unstructured_planning(mock_stream_response, mock_state):
    mock_stream_response.return_value = "reasoning_plan_result"
    result = unstructured_planning(mock_state)
    assert result == {"reasoning_plan": "reasoning_plan_result"}


@patch("self_discover._iself_discover.stream_response")
def test_unstructured_application(mock_stream_response, mock_state):
    mock_stream_response.return_value = "reasoning_result"
    result = unstructured_application(mock_state)
    assert result == {"reasoning": "reasoning_result"}


@patch("self_discover._iself_discover.StateGraph")
def test_build_iself_discover_graph(mock_state_graph):
    mock_graph_builder = MagicMock()
    mock_state_graph.return_value = mock_graph_builder

    graph = build_iself_discover_graph(structured=True)
    assert mock_graph_builder.add_node.call_count == 4
    assert mock_graph_builder.add_edge.call_count == 4

    graph = build_iself_discover_graph(structured=False)
    assert mock_graph_builder.add_node.call_count == 4
    assert mock_graph_builder.add_edge.call_count == 4


@patch("self_discover._iself_discover.invoke_graph")
@patch("self_discover._iself_discover.validate_params")
def test_iself_discover(mock_validate_params, mock_invoke_graph, mock_llm_model):
    mock_invoke_graph.return_value = {"result_key": "result_value"}
    result = iself_discover(
        task_description="Test task",
        model=mock_llm_model,
        answer_formats="Test format",
        structured=True,
        stream=False,
    )
    mock_validate_params.assert_called_once()
    assert result == {"result_key": "result_value"}

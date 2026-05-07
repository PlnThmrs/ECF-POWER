import pytest
import requests_mock


@pytest.fixture
def requests_mock_fixture():
    with requests_mock.Mocker() as m:
        yield m

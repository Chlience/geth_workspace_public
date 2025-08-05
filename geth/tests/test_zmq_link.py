import pytest

from geth.base.zmq_link import ZmqClient, ZmqServer


class CustomObject:
    def __init__(self, value):
        self.value = value


@pytest.fixture(scope="module")
def zmq_server():
    server = ZmqServer(port=5588)
    yield server


@pytest.fixture(scope="module")
def zmq_client():
    client = ZmqClient(client_id="test_client")
    client.connect("tcp://localhost:5588")
    yield client


def test_zmq_server_client_transfer_string(zmq_server, zmq_client):
    message = "Hello, World!"
    zmq_client.send(message)
    identity, received_message = zmq_server.recv()
    assert received_message == message
    zmq_server.send(identity, received_message)
    assert zmq_client.recv() == message


def test_zmq_server_client_transfer_dict(zmq_server, zmq_client):
    message = {"key": "value", "nested": {"key": "value"}}
    zmq_client.send(message)
    identity, received_message = zmq_server.recv()
    assert received_message == message
    zmq_server.send(identity, received_message)
    assert zmq_client.recv() == message


def test_zmq_server_client_transfer_list(zmq_server, zmq_client):
    message = [1, 2, 3, 4, 5]
    zmq_client.send(message)
    identity, received_message = zmq_server.recv()
    assert received_message == message
    zmq_server.send(identity, received_message)
    assert zmq_client.recv() == message


def test_zmq_server_client_transfer_tuple(zmq_server, zmq_client):
    message = (1, "two", 3.0)
    zmq_client.send(message)
    identity, received_message = zmq_server.recv()
    assert received_message == message
    zmq_server.send(identity, received_message)
    assert zmq_client.recv() == message


def test_zmq_server_client_transfer_set(zmq_server, zmq_client):
    message = {1, 2, 3, 4, 5}
    zmq_client.send(message)
    identity, received_message = zmq_server.recv()
    assert received_message == message
    zmq_server.send(identity, received_message)
    assert zmq_client.recv() == message


def test_zmq_server_client_transfer_custom_object(zmq_server, zmq_client):
    message = CustomObject(42)
    zmq_client.send(message)
    identity, received_message = zmq_server.recv()
    assert isinstance(received_message, CustomObject)
    assert received_message.value == message.value
    zmq_server.send(identity, received_message)
    received_message = zmq_client.recv()
    assert isinstance(received_message, CustomObject)
    assert received_message.value == message.value

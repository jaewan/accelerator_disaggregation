import zmq
import torch
import io

def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print("Server started, waiting for requests...")

    while True:
        # Wait for next request from client
        message = socket.recv()
        
        # Check if it's a tensor or an operation
        if message == b"add":
            print("Received 'add' operation request")
            # Receive tensors
            tensor_a_msg = socket.recv()
            tensor_b_msg = socket.recv()

            # Deserialize tensors
            tensor_a = torch.load(io.BytesIO(tensor_a_msg))
            tensor_b = torch.load(io.BytesIO(tensor_b_msg))
            print(f"Received tensor a: {tensor_a}")
            print(f"Received tensor b: {tensor_b}")
            # Perform the operation on the GPU
            with torch.cuda.device(0):  # Assuming you have a GPU at index 0
                result_tensor = torch.add(tensor_a.to("cuda"), tensor_b.to("cuda"))
            print(f"Result tensor: {result_tensor}")
            # Serialize the result tensor
            buffer = io.BytesIO()
            torch.save(result_tensor.to("cpu"), buffer)
            result_tensor_msg = buffer.getvalue()

            # Send the result tensor back
            socket.send(result_tensor_msg)
        else:
            print("Error: Unknown operation requested")
            socket.send(b"ERROR: Unknown operation")

if __name__ == "__main__":
    run_server()

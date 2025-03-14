## Stitching distributed futures with transparent custom device extension
### Core Question:Can a Custom Device Extension Return a Future?

The short answer is: No, not directly, and it presents a significant challenge.  Here's why and what to consider:

- **PyTorch's Dispatcher and Tensor API Expectations**: PyTorch's dispatcher and tensor operations (like matmul) are designed to work synchronously and expect immediate Tensor results.  Returning a Future (or a promise of a future tensor) directly violates this fundamental expectation.  The dispatcher doesn't know what to do with a Future.  It needs a Tensor to continue the computational graph, register gradients, and so on.


### Potential solution: The "Proxy Tensor" 

The key is to make your custom device extension appear to return a standard Tensor while internally managing the distributed future.  This is where the concept of a "Proxy Tensor" comes into play.

1. Proxy Tensor:
	- Is a subclass of torch.Tensor: It must behave like a regular tensor from the perspective of PyTorch's dispatcher and other operations. It needs to pass all the type checks and have the necessary methods.
	- Holds a future internally: The proxy tensor contains a reference to the future object reporesenting the result on the remote accelerator.

2. Overriding Methods:

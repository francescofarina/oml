# OML
I'm starting OML with the aim to be a Rust framework for Online Machine Learning, 
allowing concurrent training and inference on arbitrary models. 

Most ML frameworks requires to first train a model, deploy it and then run inference steps
on a static model. OML explores how to safely allow one (and only one) process to update (write)
the weights of a model, while allowing multiple other processes to read the same weights to 
concurrently perform inference steps. Ideally, all of that should be asynchronous so that
read events can occur at any time, even in the middle of a write event (that's not a problem
as it's basically equivalent of using the weights updated by a block gradient descent algorithm).

> [!WARNING]  
> I've just started developing this library - currently mainly exploring concurrency rather than ML functionalities. Any contribution is welcome.

## Quickstart
Clone the repository and build it (`cargo build`).

### Current structure
- `model.rs` contains a basic definition of a model (just a collection of parameters) implementing two methods for initializing it
- `algorithms.rs` contains traits to implement algorithms (each having methods for training steps and inference steps)
- `handlers.rs` provides handlers to gather input data and interact with the model methods
- `server.rs` provides a basic serve implementation exposing the two endpoints for training and inference
- `tensors.rs` currently contains just a skeleton tensor implementation and is unused
- `main.rs` contains a working example that can be run via `cargo run` 

## TODO
- [ ] check whether it's possible to directly use an external framework such as Burn to build models (there may be issues in how parameters and backprop graph are handled that prevents from concurrently running training and inference steps)
- [ ] implement basic example with recursive least squares
- [ ] iplement some safety procedures - right now things work under the assumption that only one write thread is active at a time but that would need to be enforced somehow


## Testing
To test the various modules run `cargo test`.

To test the ability to concurrently perform training and inference steps you can:
- run the script in `scripts/test.sh` (though that doesn't necessarily show concurrency right now)
- manually test:
  - run `cargo run` to launch the server
  - open two terminals
    - in the first one run `curl -X POST http://localhost:8080/training -H "Content-Type: application/json" -d '10'` (the training as a simulated duration of 5 seconds)
    - in the second one run multiple times `curl -X POST http://localhost:8080/inference -H "Content-Type: application/json" -d '3.5'` (each inference as a simulated duration of 0.5 seconds)
  - if you are able to run multiple inference steps while the training one is running it means that concurrency is working properly

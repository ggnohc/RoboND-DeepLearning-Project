## Project: Follow Me
---

### The write-up conveys the an understanding of the network architecture.

##### The student clearly explains each layer of the network architecture and the role that it plays in the overall network. The student can demonstrate the benefits and/or drawbacks different network architectures pertaining to this project and can justify the current network with factual data. Any choice of configurable parameters should also be explained in the network architecture.

##### The student shall also provide a graph, table, diagram, illustration or figure for the overall network to serve as a reference for the reviewer.

  * The network contains 5 layers as follows:
    * 2 encoder layer with 32 and 64 layers respectively
    * 1 1x1 convolution layer
    * 2 decoder layer with 64 and 32 layers respectively, to match the encoder layers

    ```
    ####  ENCODER LAYER ####
    encoder_layer_1 = encoder_block(inputs, filters=32, strides=2)
    encoder_layer_2 = encoder_block(encoder_layer_1, filters=64, strides=2)

    #### 1x1 CONVULUTION LAYER ####
    conv_layer = conv2d_batchnorm(encoder_layer_2, 128, kernel_size=1, strides=1)

    #### DECODER LAYER ####
    decoder_layer_2 = decoder_block(conv_layer, encoder_layer_1, filters=64)
    decoder_layer_1 = decoder_block(decoder_layer_2, inputs, filters=32)

    ```

### The write-up conveys the student's understanding of the parameters chosen for the the neural network.

##### The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.) Hyper parameters include, but are not limited to:

      Epoch
      Learning Rate
      Batch Size
      Etc.

  All configurable parameters should be explicitly stated and justified.     

### The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.

### The student is demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.

### The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.

### The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

##### The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

### The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

##### The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.

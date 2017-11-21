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
  *  The choice of relatively simple 5 layer is because we are training on a medium resolution image (256 x 256 pixels for both height and width), adding more layers will add unneeded complexity and increase computation time during training.

      * Below is output from "model.summary()"
        ```
        Layer (type)                 Output Shape              Param #   
        =================================================================
        input_5 (InputLayer)         (None, 160, 160, 3)       0         
        _________________________________________________________________
        separable_conv2d_keras_17 (S (None, 80, 80, 32)        155       
        _________________________________________________________________
        batch_normalization_21 (Batc (None, 80, 80, 32)        128       
        _________________________________________________________________
        separable_conv2d_keras_18 (S (None, 40, 40, 64)        2400      
        _________________________________________________________________
        batch_normalization_22 (Batc (None, 40, 40, 64)        256       
        _________________________________________________________________
        conv2d_9 (Conv2D)            (None, 40, 40, 128)       8320      
        _________________________________________________________________
        batch_normalization_23 (Batc (None, 40, 40, 128)       512       
        _________________________________________________________________
        bilinear_up_sampling2d_9 (Bi (None, 80, 80, 128)       0         
        _________________________________________________________________
        concatenate_9 (Concatenate)  (None, 80, 80, 160)       0         
        _________________________________________________________________
        separable_conv2d_keras_19 (S (None, 80, 80, 64)        11744     
        _________________________________________________________________
        batch_normalization_24 (Batc (None, 80, 80, 64)        256       
        _________________________________________________________________
        bilinear_up_sampling2d_10 (B (None, 160, 160, 64)      0         
        _________________________________________________________________
        concatenate_10 (Concatenate) (None, 160, 160, 67)      0         
        _________________________________________________________________
        separable_conv2d_keras_20 (S (None, 160, 160, 32)      2779      
        _________________________________________________________________
        batch_normalization_25 (Batc (None, 160, 160, 32)      128       
        _________________________________________________________________
        conv2d_10 (Conv2D)           (None, 160, 160, 3)       99        
        =================================================================
        Total params: 26,777
        Trainable params: 26,137
        Non-trainable params: 640
        ```



### The write-up conveys the student's understanding of the parameters chosen for the the neural network.

##### The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.) Hyper parameters include, but are not limited to:

  * learning_rate = 0.002
    * A learning rate of 0.002 is chosen as it is the recommended learning rate for Nadam, have tried with lower learning of 0.001 but the improvement is insignificant.  In general found that learning rate in magnitude of 1/1000 is optimal
  * num_epochs = 20
  * batch_size = 32
    * Tbe batch size is normally power of 2, since I found it most convenient to train locally on my laptop with a Nvidia 1050 GPU, and it is the max batch size before OOM (out of memory) error occur, I have stick with it.
    * Have tried training with batch size of 128 on AWS, but it has ended with lower final score
  * steps_per_epoch = number_of_training_images // batch_size
    * I set the steps per epoch to be number of training image divided by batch size as recommended, it is close to the default value so I have stick with it.
  * validation_steps = number_of_validation_images // batch_size
    * I set validation step similar to steps per epoch
  * workers = 2     
    * Setting workers to 2 as I have a lower end GPU
    *  Have tried setting to much higher value of 32 when trained on GPU, but did not observe significant improvement.

### The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.

*  Collected more data with Hero
*  Switch from CPU training to GPU training

### The student is demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.

### The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.

### The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

##### The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

### The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

##### The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.

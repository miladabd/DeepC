# DeepC
Implementing Deep Convolutional Neural Network in C without External Libraries for YUV video Super-Resolution

This code uses [FSRCNN](https://arxiv.org/abs/1608.00367) algorithm to upsample the Y component of each frame of the YUV 4:2:0 video with scale factor of 2.

![framework](https://user-images.githubusercontent.com/29326313/133723014-248f3acd-638d-4e45-834c-88ab8f5ad801.png)

No external liberaries are used and all required functions (including convolution, deconvolution and non-linearities) are written in main file ```source.c``` file. 

To compile the code in Linux use:

``` gcc source.c -o videosr```

Then run the code using following command:

```./videosr <input_video_name> <output_video_name>```

For example, in order to upsample ```foreman_qcif_146x144.yuv```:

```./videosr foreman_qcif_146x144.yuv output_foreman_352x288.yuv```

module conv2d(
    output y,
    input weight,
    input bias,
    input width, 
    input height,
    input in_channels,
    input out_channels,
    input ksize
);

reg y_tmp;

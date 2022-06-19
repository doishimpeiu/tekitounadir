module count3 (
    output reg [1:0] q,
    output max,
    input enable,
    input clk,
    input xrst
);
assign out_channels = (enable && och == 3) ? 1 : 0;
assign height = (enable && h == 27) ? 1 : 0;
assign width = (enable && w == 27) ? 1 : 0;
assign in_channels = (enable && ich == 0) ? 1 : 0;
assign ksize_height = (enable && kh == 2) ? 1 : 0;
assign ksize_weight = (enable && kw == 2) ? 1 : 0;
always @(posedge clk or negedge xrst) begin

    ph = (h + kh - int(ksize/2))
    pw = (w + kw - int(ksize/2))
    if (ph < 0 or ph >= height or pw < 0 or pw >= width):
        continue
    pix_idx = ((ich * height + ph) * width + pw)
    weight_idx = (((och * in_channels + ich) * ksize + kh) * ksize + kw)
    sum += x[pix_idx] * weight[weight_idx]
    if (!xrst) begin
        kw <= 0;
    end
    else if (max) begin
        kw <= 0;
    end
    else if (enable) begin
        kw <= kw + 1;
    end

    if (!xrst) begin
        kh <= 0;
    end
    else if (max) begin
        kh <= 0;
    end
    else if (enable) begin
        kh <= kh + 1;
    end

    //sumの場所ここでいいか確認
    sum = 0
    if (!xrst) begin
        ich <= 0;
    end
    else if (max) begin
        ich <= 0;
    end
    else if (enable) begin
        ich <= ich + 1;
    end

    sum += bias[och]
    y_tmp[(och * height + h) * width + w] = sum
    if (!xrst) begin
        w <= 0;
    end
    else if (max) begin
        w <= 0;
    end
    else if (enable) begin
        w <= w + 1;
    end

    if (!xrst) begin
        h <= 0;
    end
    else if (max) begin
        h <= 0;
    end
    else if (enable) begin
        h <= h + 1;
    end

    if (!xrst) begin
        och <= 0;
    end
    else if (max) begin
        och <= 0;
    end
    else if (enable) begin
        och <= och + 1;
    end
    
    if (!xrst) begin
        q <= 0;
    end
    else if (max) begin
        q <= 0;
    end
    else if (enable) begin
        q <= q + 1;
    end
y = y_tmp.reshape(out_channels, width, height)
end
endmodule
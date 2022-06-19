module count3 (
    output reg [1:0] q,
    output max,
    input enable,
    input clk,
    input xrst
);
assign max = (enable && q == 2) ? 1 : 0;
always @(posedge clk or negedge xrst) begin
    if (!xrst) begin
        q <= 0;
    end
    else if (max) begin
        q <= 0;
    end
    else if (enable) begin
        q <= q + 1;
    end
end
endmodule

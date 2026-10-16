// Earliest supplied clock wins; equal due times preserve input order.
// Clock generation (exp/log/randomness) and candidate discovery are upstream.
module sm_race_select #(
  parameter integer TIME_W=64, ADDR_W=16, TAG_W=64
)(
  input wire clk, rst_n,
  input wire in_valid, output wire in_ready,
  input wire in_last,
  input wire [TIME_W-1:0] in_due,
  input wire [ADDR_W-1:0] in_addr,
  input wire [TAG_W-1:0] in_tag,
  output reg out_valid, input wire out_ready,
  output reg [TIME_W-1:0] out_due,
  output reg [ADDR_W-1:0] out_addr,
  output reg [TAG_W-1:0] out_tag
);
  reg have_best;
  reg [TIME_W-1:0] best_due;
  reg [ADDR_W-1:0] best_addr;
  reg [TAG_W-1:0] best_tag;
  wire replace_best = !have_best || in_due < best_due;
  assign in_ready = !out_valid;
  always @(posedge clk) begin
    if (!rst_n) begin
      have_best<=0; out_valid<=0;
      best_due<=0;best_addr<=0;best_tag<=0;
      out_due<=0;out_addr<=0;out_tag<=0;
    end else begin
      if(out_valid && out_ready) out_valid<=0;
      if(in_valid && in_ready) begin
        if(in_last) begin
          out_valid<=1;have_best<=0;
          out_due<=replace_best ? in_due : best_due;
          out_addr<=replace_best ? in_addr : best_addr;
          out_tag<=replace_best ? in_tag : best_tag;
        end else begin
          have_best<=1;
          if(replace_best) begin best_due<=in_due;best_addr<=in_addr;best_tag<=in_tag;end
        end
      end
    end
  end
endmodule

// Lazy complex temporal transport. Upstream supplies exp(-rate*age)*cos/sin.
// Signed Q format, arithmetic shift (floor), saturation. One complex pair/cycle.
// No claim that supplied coefficients are computed in this block.
module sm_temporal_transport #(
  parameter integer PAIRS=4, W=24, FRAC=16,
  parameter integer INDEX_W=(PAIRS<=1 ? 1 : $clog2(PAIRS))
)(
 input wire clk,rst_n,input wire in_valid,output wire in_ready,
 input wire [PAIRS*2*W-1:0] in_values,
 input wire [PAIRS*W-1:0] in_cos_decay,in_sin_decay,
 output wire out_valid,input wire out_ready,
 output reg [PAIRS*2*W-1:0] out_values,output reg out_saturated
);
 localparam IDLE=0,CALC=1,RESULT=2;
 reg [1:0] state;reg [INDEX_W-1:0] index;
 reg [PAIRS*2*W-1:0] values;
 reg [PAIRS*W-1:0] cs,ss;
 wire signed [W-1:0] xr=$signed(values[(2*index)*W +: W]);
 wire signed [W-1:0] xi=$signed(values[(2*index+1)*W +: W]);
 wire signed [W-1:0] c=$signed(cs[index*W +: W]);
 wire signed [W-1:0] s=$signed(ss[index*W +: W]);
 wire signed [2*W-1:0] rc=xr*c,is_=xi*s,rs=xr*s,ic=xi*c;
 wire signed [2*W:0] real_wide=$signed({rc[2*W-1],rc})-$signed({is_[2*W-1],is_});
 wire signed [2*W:0] imag_wide=$signed({rs[2*W-1],rs})+$signed({ic[2*W-1],ic});
 wire signed [2*W:0] real_scaled=real_wide >>> FRAC,imag_scaled=imag_wide >>> FRAC;
 localparam signed [2*W:0] HIGH=( ({{(2*W){1'b0}},1'b1} << (W-1))-1 );
 localparam signed [2*W:0] LOW=-( {{(2*W){1'b0}},1'b1} << (W-1) );
 function automatic [W-1:0] saturate(input signed [2*W:0] x);
   begin if(x>HIGH)saturate={1'b0,{(W-1){1'b1}}};else if(x<LOW)saturate={1'b1,{(W-1){1'b0}}};else saturate=x[W-1:0];end
 endfunction
 assign in_ready=state==IDLE;assign out_valid=state==RESULT;
 always @(posedge clk) begin
   if(!rst_n)begin state<=IDLE;index<=0;out_values<=0;out_saturated<=0;values<=0;cs<=0;ss<=0;end
   else case(state)
    IDLE:if(in_valid)begin values<=in_values;cs<=in_cos_decay;ss<=in_sin_decay;index<=0;out_saturated<=0;state<=CALC;end
    CALC:begin
      out_values[(2*index)*W +: W]<=saturate(real_scaled);
      out_values[(2*index+1)*W +: W]<=saturate(imag_scaled);
      out_saturated<=out_saturated || real_scaled>HIGH || real_scaled<LOW || imag_scaled>HIGH || imag_scaled<LOW;
      if(index==PAIRS-1)state<=RESULT;else index<=index+1'b1;
    end
    RESULT:if(out_ready)state<=IDLE;
    default:state<=IDLE;
   endcase
 end
endmodule

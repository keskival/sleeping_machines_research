// One addressed command port. No idle payload sweep. Timestamps never wrap.
// Status: 0 success, 1 empty, 2 stale version, 3 past time, 4 version overflow,
// 5 invalid address. Empty state expects version zero; first commit is version 1.
module sm_versioned_state_bank #(
  parameter integer ENTRIES=64, ADDR_W=6, DATA_W=256, TIME_W=64, VERSION_W=32
)(
  input wire clk,rst_n,
  input wire cmd_valid,output wire cmd_ready,input wire cmd_write,
  input wire [ADDR_W-1:0] cmd_addr,
  input wire [TIME_W-1:0] cmd_time,
  input wire [VERSION_W-1:0] cmd_expected,
  input wire [DATA_W-1:0] cmd_data,
  output reg rsp_valid,input wire rsp_ready,
  output reg [2:0] rsp_status,
  output reg [DATA_W-1:0] rsp_data,
  output reg [TIME_W-1:0] rsp_time,
  output reg [VERSION_W-1:0] rsp_version
);
  reg [DATA_W-1:0] values[0:ENTRIES-1];
  reg [TIME_W-1:0] times[0:ENTRIES-1];
  reg [VERSION_W-1:0] versions[0:ENTRIES-1];
  reg [ENTRIES-1:0] valid;
  assign cmd_ready=!rsp_valid;
  always @(posedge clk) begin
    if(!rst_n) begin valid<=0;rsp_valid<=0;rsp_status<=0;rsp_data<=0;rsp_time<=0;rsp_version<=0;end
    else begin
      if(rsp_valid && rsp_ready)rsp_valid<=0;
      if(cmd_valid && cmd_ready) begin
        rsp_valid<=1;rsp_data<=0;rsp_time<=0;rsp_version<=0;rsp_status<=0;
        if(cmd_addr>=ENTRIES)rsp_status<=5;
        else if(cmd_write) begin
          if((!valid[cmd_addr] && cmd_expected!=0) || (valid[cmd_addr] && versions[cmd_addr]!=cmd_expected))rsp_status<=2;
          else if(valid[cmd_addr] && cmd_time<times[cmd_addr])rsp_status<=3;
          else if(valid[cmd_addr] && (&versions[cmd_addr]))rsp_status<=4;
          else begin
            values[cmd_addr]<=cmd_data;times[cmd_addr]<=cmd_time;
            versions[cmd_addr]<=cmd_expected+1'b1;valid[cmd_addr]<=1;
            rsp_data<=cmd_data;rsp_time<=cmd_time;rsp_version<=cmd_expected+1'b1;
          end
        end else if(!valid[cmd_addr])rsp_status<=1;
        else if(cmd_time<times[cmd_addr])rsp_status<=3;
        else begin rsp_data<=values[cmd_addr];rsp_time<=times[cmd_addr];rsp_version<=versions[cmd_addr];end
      end
    end
  end
endmodule

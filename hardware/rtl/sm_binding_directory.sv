// Bounded immutable parent-version -> child-version bindings. Separate child
// identity and pose: children sharing a pose never alias. Snapshot values live
// in a separate owner-managed store; this block holds references only.
module sm_binding_directory #(
 parameter integer ENTRIES=16, ID_W=16, VERSION_W=32, TIME_W=64, POSE_W=32,
 parameter integer SLOT_W=(ENTRIES<=1 ? 1 : $clog2(ENTRIES))
)(
 input wire clk,rst_n,
 input wire put_valid,output wire put_ready,
 input wire [ID_W-1:0] put_parent,put_child,
 input wire [VERSION_W-1:0] put_parent_version,put_child_version,
 input wire [TIME_W-1:0] put_child_available,put_issued,put_parent_available,
 input wire [POSE_W-1:0] put_pose,
 output reg ack_valid,input wire ack_ready,output reg [2:0] ack_status,
 output reg [SLOT_W-1:0] ack_slot,
 input wire get_valid,output wire get_ready,input wire [SLOT_W-1:0] get_slot,
 input wire [TIME_W-1:0] get_cutoff,
 output reg rsp_valid,input wire rsp_ready,output reg [2:0] rsp_status,
 output reg [ID_W-1:0] rsp_parent,rsp_child,
 output reg [VERSION_W-1:0] rsp_parent_version,rsp_child_version,
 output reg [TIME_W-1:0] rsp_issued,output reg [POSE_W-1:0] rsp_pose
);
 reg [ENTRIES-1:0] valid;
 reg [ID_W-1:0] parents[0:ENTRIES-1],children[0:ENTRIES-1];
 reg [VERSION_W-1:0] parent_versions[0:ENTRIES-1],child_versions[0:ENTRIES-1];
 reg [TIME_W-1:0] child_ready[0:ENTRIES-1],issued[0:ENTRIES-1],parent_ready[0:ENTRIES-1];
 reg [POSE_W-1:0] poses[0:ENTRIES-1];
 integer i,match_slot,free_slot;
 always @* begin
  match_slot=-1;free_slot=-1;
  for(i=0;i<ENTRIES;i=i+1) begin
   if(!valid[i] && free_slot<0)free_slot=i;
   if(valid[i] && parents[i]==put_parent && parent_versions[i]==put_parent_version && children[i]==put_child)match_slot=i;
  end
 end
 // Single serialized directory port. Reads win when both are presented.
 assign get_ready=!rsp_valid && !ack_valid;
 assign put_ready=!rsp_valid && !ack_valid && !get_valid;
 always @(posedge clk) begin
  if(!rst_n) begin valid<=0;ack_valid<=0;ack_status<=0;ack_slot<=0;rsp_valid<=0;rsp_status<=0;rsp_parent<=0;rsp_child<=0;rsp_parent_version<=0;rsp_child_version<=0;rsp_issued<=0;rsp_pose<=0;end
  else begin
   if(ack_valid && ack_ready)ack_valid<=0;
   if(rsp_valid && rsp_ready)rsp_valid<=0;
   if(get_valid && get_ready)begin
    rsp_valid<=1;rsp_status<=0;rsp_parent<=0;rsp_child<=0;rsp_parent_version<=0;rsp_child_version<=0;rsp_issued<=0;rsp_pose<=0;
    if(get_slot>=ENTRIES || !valid[get_slot])rsp_status<=1;
    else if(get_cutoff<parent_ready[get_slot])rsp_status<=3;
    else begin rsp_parent<=parents[get_slot];rsp_child<=children[get_slot];rsp_parent_version<=parent_versions[get_slot];rsp_child_version<=child_versions[get_slot];rsp_issued<=issued[get_slot];rsp_pose<=poses[get_slot];end
   end
   if(put_valid && put_ready)begin
    ack_valid<=1;ack_status<=0;ack_slot<=0;
    if(put_child_available>put_issued || put_issued>put_parent_available)ack_status<=3;
    else if(match_slot>=0)begin
     ack_slot<=match_slot;
     if(child_versions[match_slot]!=put_child_version || issued[match_slot]!=put_issued || poses[match_slot]!=put_pose || child_ready[match_slot]!=put_child_available || parent_ready[match_slot]!=put_parent_available)ack_status<=2;
    end else if(free_slot<0)ack_status<=4;
    else begin
     ack_slot<=free_slot;valid[free_slot]<=1;parents[free_slot]<=put_parent;parent_versions[free_slot]<=put_parent_version;
     children[free_slot]<=put_child;child_versions[free_slot]<=put_child_version;child_ready[free_slot]<=put_child_available;
     issued[free_slot]<=put_issued;parent_ready[free_slot]<=put_parent_available;poses[free_slot]<=put_pose;
    end
   end
  end
 end
endmodule

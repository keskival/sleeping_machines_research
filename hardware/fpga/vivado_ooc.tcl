# Run: SM_FPGA_PART=<exact part> SM_TOP=sm_race_select SM_CLOCK_NS=10 \
#      vivado -mode batch -source hardware/fpga/vivado_ooc.tcl
# OOC block implementation, NOT a board bitstream or full-model accelerator.
foreach key {SM_FPGA_PART SM_TOP SM_CLOCK_NS} {
    if {![info exists ::env($key)]} { error "Set $key explicitly" }
}
set top $::env(SM_TOP)
set allowed {sm_race_select sm_versioned_state_bank sm_temporal_transport sm_binding_directory}
if {$top ni $allowed} {error "Unknown draft top"}
set root [file normalize [file join [file dirname [info script]] ../..]]
set output [file join $root .git hardware-vivado $top]
file mkdir $output
create_project -in_memory -part $::env(SM_FPGA_PART)
read_verilog -sv [file join $root hardware rtl ${top}.sv]
synth_design -mode out_of_context -top $top -part $::env(SM_FPGA_PART)
create_clock -period $::env(SM_CLOCK_NS) [get_ports clk]
report_utilization -hierarchical -file [file join $output utilization_synth.txt]
opt_design
place_design
phys_opt_design
route_design
report_utilization -hierarchical -file [file join $output utilization_route.txt]
report_timing_summary -file [file join $output timing_route.txt]
# Vectorless tool estimate until representative activity is annotated.
report_power -file [file join $output power_estimate_vectorless.txt]
write_checkpoint -force [file join $output routed.dcp]

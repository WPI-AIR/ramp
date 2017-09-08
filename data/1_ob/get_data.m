%--------------------------------------------------------------------------
%                           1 Obstacle data
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% General data
%--------------------------------------------------------------------------
compute_switch_ts = importdata('compute_switch_ts.txt');
full_trajec = importdata('full_trajectory.txt');
min_dist_obs = importdata('min_dist_obs.txt');
motion_error_amount = importdata('motion_error_amount.txt');
num_ccs = importdata('num_ccs.txt');
num_pcs = importdata('num_pcs.txt');
num_scs = importdata('num_scs.txt');
pc_freqs = importdata('pc_freqs.txt');
cc_freqs = importdata('cc_freqs.txt');
sc_freqs = importdata('sc_freqs.txt');
pop_size = importdata('pop_size.txt');
runtime = importdata('runtime.txt');
switch_t_size = importdata('switch_t_size.txt');
time_in_ic = importdata('time_in_ic.txt');
trajec_size = importdata('trajec_size.txt');


%--------------------------------------------------------------------------
% Duration data
%--------------------------------------------------------------------------
cc_durs = importdata('cc_durs.txt');
pc_durs = importdata('pc_durs.txt');
sc_durs = importdata('sc_durs.txt');
durs_path_mods = importdata('durations_path_mods.txt');
durs_sensing = importdata('durations_sensing.txt');
durs_trj_eval = importdata('durations_trj_eval.txt');
durs_trj_gen = importdata('durations_trj_gen.txt');
error_correct_durs_eval = importdata('error_correct_durs_eval.txt');
error_correct_durs_no_eval = importdata('error_correct_durs_no_eval.txt');
eval_durs = importdata('eval_durs.txt');
trajec_durs = importdata('trajec_durs.txt');
mod_durs = importdata('mod_durs.txt');
mutate_durs = importdata('mutate_durs.txt');

load("C:\Users\Fhlin\Documents\GitHub\APS360-T39\MATLAB\data_processing\fconn_rest_hcp_yeo17network_aseg_mc.mat");

num_subjects = length(fconn);  % Should be 1113
num_networks = 17;  % Networks 2 to 18
run_idx = 1;

reconstructed_data = cell(1, num_subjects);  % Store final result for all subjects

for subj = 1:num_subjects
    subject_struct = fconn{subj};  % 1x1 struct
    if isempty(subject_struct) || ~isfield(subject_struct, 'fconn') || isempty(subject_struct.fconn)
        fprintf('Skipping subject %d: no fconn\n', subj);
        continue;
    end
    networks = subject_struct.fconn;  % 4x18 cell array
    subject_result = cell(1, num_networks);  % To store 17 reconstructed matrices
    
    for net = 1:num_networks
        net_idx = net + 1;  % Network 2 to 18
        net_data = [];

        for r = 1:4
            temp = networks{r, net_idx};
            if iscell(temp)
                temp = temp{1};
            end
            if ~isempty(temp)
                net_data = temp;
                break;  
            end
        end
        
        U = net_data.fconn_u10;
        S = net_data.fconn_s10;
        subject_result{net} = U * diag(S);
       
    end

    reconstructed_data{subj} = subject_result;
end

output_dir = './output';
num_subjects = length(reconstructed_data);

for subj = 1:num_subjects
    subject_entry = reconstructed_data{subj};
    
    if isempty(subject_entry) || ~isfield(subject_entry, 'networks')
        fprintf('Skipping subject %d (no network data)\n', subj);
        continue;
    end
    
    subject_id = subject_entry.subject_id;
    subject_networks = subject_entry.networks;
    num_networks = length(subject_networks);

    % Create folder named with subject ID
    subj_folder = fullfile(output_dir, sprintf('subject_%s', subject_id));
    if ~exist(subj_folder, 'dir')
        mkdir(subj_folder);
    end

    % Optional: Save subject ID to a text file
    fid = fopen(fullfile(subj_folder, 'subject_id.txt'), 'w');
    fprintf(fid, '%s\n', subject_id);
    fclose(fid);

    % Save each network as a .npy file labeled net_01.npy to net_17.npy
    for net = 1:num_networks
        data = subject_networks{net};
        if isempty(data)
            continue;
        end
        filename = sprintf('net_%02d.npy', net);  % Now: net_01 to net_17
        filepath = fullfile(subj_folder, filename);
        writeNPY(data, filepath);
    end
end

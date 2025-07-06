output_dir = './output';  % e.g., '/mnt/project/processed_data'
num_subjects = length(reconstructed_data);
num_networks = length(reconstructed_data{2});  % assuming 2nd subject has all networks

for subj = 2:num_subjects  % assuming you skipped subj==1
    subj_folder = fullfile(output_dir, sprintf('subj_%04d', subj));
    if ~exist(subj_folder, 'dir')
        mkdir(subj_folder);
    end

    if isempty(reconstructed_data{subj})
        fprintf('Skipping subject %d (no network data)\n', subj);
        continue;
    end

    for net = 1:num_networks
        data = reconstructed_data{subj}{net};
        filename = sprintf('net_%02d.npy', net + 1);  % offset by 1 if networks 2 to 18
        filepath = fullfile(subj_folder, filename);
        writeNPY(data, filepath);
    end
end

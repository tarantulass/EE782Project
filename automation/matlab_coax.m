OUTPUT_DIR = "PATCH_10K";
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

FREQ_RANGE = linspace(1e9, 5e9, 2e4);   
p1 = patchMicrostrip;  % Changed to coaxial feed patch antenna
p1.Substrate = dielectric(Name="FR4", EpsilonR=4.8);

Results = cell(numel(FREQ_RANGE), 1);

for k = 1:numel(FREQ_RANGE)
    freq = FREQ_RANGE(k);

    % Design antenna at this frequency
    ant = design(p1, freq);

    % Extract antenna geometric parameters
    patchLength = ant.Length;
    patchWidth = ant.Width;
    patchHeight = ant.Height;
    feedPosX = ant.FeedOffset(1); % Feed X offset
    feedPosY = ant.FeedOffset(2); % Feed Y offset

    substrateName = string(ant.Substrate.Name);
    er = ant.Substrate.EpsilonR;

    % Store all data in one table row
    Results{k} = table(freq, ...
                       patchLength, patchWidth, patchHeight, feedPosX, feedPosY, ...
                       substrateName, er, ...
                       'VariableNames', {'Freq_Hz', ...
                                         'PatchLength', 'PatchWidth', 'PatchHeight', 'FeedPosX', 'FeedPosY', ...
                                         'Substrate', 'EpsilonR'});
end

AllResults = vertcat(Results{:});
writetable(AllResults, fullfile(OUTPUT_DIR, 'patch_data_coaxial.csv'));
disp("Data saved successfully to 'PATCH_10K/patch_data_coaxial.csv'");

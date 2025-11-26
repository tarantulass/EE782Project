%% OUTPUT DIRECTORY
OUTPUT_DIR = "PATCH_10K";
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

%% FREQUENCY RANGE
FREQ_RANGE = linspace(0.1e9, 1e9, 2e4);   
p1 = patchMicrostripInsetfed;
p1.Substrate = dielectric(Name="FR4",EpsilonR=4.8);

%% MAIN LOOP WITH WAITBAR
Results = cell(numel(FREQ_RANGE),1);

for k = 1:numel(FREQ_RANGE)
    freq = FREQ_RANGE(k);

    % Design antenna at this frequency
    ant = design(p1, freq);

    % Extract antenna geometric parameters
    patchLength   = ant.Length;
    patchWidth    = ant.Width;
    patchheight    = ant.Height;
    feedOffset    = ant.FeedOffset(1);

    stripline = ant.StripLineWidth;
    notchlength = ant.NotchLength;
    notchwidth = ant.NotchWidth;
    gndlength = ant.GroundPlaneLength;
    gndwidth = ant.GroundPlaneWidth;
    
    substrateName = string(ant.Substrate.Name);
    er            = ant.Substrate.EpsilonR;

    % % Compute electrical properties
    %% Uncomment to compute S-parameters and impedance respectively
    %% Compute will take a lot of time so reduce the number of frequencies to be observed 
    % s = sparameters(ant, FREQ_RANGE);
    % imp = impedance(ant, FREQ_RANGE);
    % fprintf(s)
    % Store all data in one table row
    Results{k} = table(freq, ...
                       patchLength, patchWidth, patchheight, stripline , feedOffset, notchlength, notchwidth, gndlength, gndwidth, ...
                       substrateName, er, ...
                       'VariableNames', {'Freq_Hz', ...
                                         'PatchLength','PatchWidth','PatchHeight','Striplinewidth','FeedOffset', 'Notchlength','Notchwidth','Gndlength','Gndwidth',...
                                         'Substrate','EpsilonR'});

end


%% COMBINE & SAVE
AllResults = vertcat(Results{:});
writetable(AllResults, fullfile(OUTPUT_DIR,'patch_data_less.csv'));
disp(" Data saved successfully to 'PATCH_10K/patch_data.csv'");
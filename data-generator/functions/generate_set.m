function [] = generate_set(csv_dir, sub_set, corpora_dir, out_dir)
fs = 16e3;
csv_name = fullfile(csv_dir, [sub_set, '.csv']);
if strcmp(sub_set, 'test_real')
    rooms       = {'cafeteria', 'courtyard'};
    real_noises      = cell(3,1);
    real_noise_files = {fullfile(corpora_dir, 'Noise', 'ambient_sound_cafeteria','cafeteria_1_10-37_min_in-ear.wav'),...
        fullfile(corpora_dir, 'Noise', 'ambient_sound_cafeteria','cafeteria_babble_1_3-23_min_in-ear.wav'),...
        fullfile(corpora_dir, 'Noise', 'ambient_sound_courtyard','courtyard_1_23-31_min_in-ear.wav')};
    for nn = 1:length(real_noise_files)
        [real_noises{nn}, fs_in] = audioread(real_noise_files{nn});
        if fs_in ~= fs, real_noises{nn} = resample(real_noises{nn}, fs, fs_in); end
    end
    n_positions    = 6;
    n_orientations = 2;
    rirs           = cell(n_positions,length(rooms), n_orientations);
    rir_dir        = fullfile(corpora_dir, 'RIR', 'HRIR_database_wav', 'hrir');
    for rr = 1:length(rooms)
        for pp = 1:n_positions
            for oo = 1:n_orientations
                [rirs{pp, rr, oo}, fs_in] = audioread(fullfile(rir_dir, rooms{rr},[rooms{rr},'_', num2str(oo), '_' , char(64 + pp), '.wav']));
                rirs{pp, rr, oo} = rirs{pp, rr, oo}(:, 1:2);
                if fs_in ~= fs
                    rirs{pp, rr, oo} = resample(rirs{pp, rr, oo}, fs, fs_in);
                end
            end
        end
    end
    DC = [];
else
    rir_dir         = fullfile(corpora_dir,   'RIR', 'AIR_1_4');
    %% Prepare noise field coherence (spherical):
    d       = 0.2; % Inter sensor distance (m)
    c       = 340; % Sound velocity (m/s)
    n_fft   = 512;
    ww      = 2*pi*fs*(0:n_fft/2)/n_fft;
    DC      = zeros(2, 2, n_fft/2+1);
    for p = 1:2
        for q = 1:2
            if p == q
                DC(p,q,:) = ones(1,1,n_fft/2+1);
            else
                DC(p,q,:) = sinc(ww*abs(p-q)*d/(c*pi));
            end
        end
    end
    rirs = [];
    real_noises = [];
    real_noise_files = [];
end

%% Start Generating files:
C = readtable(csv_name);
parfor ss = 1: size(C, 1)
    generate_signal(C(ss, :), corpora_dir, out_dir, rir_dir, sub_set, fs, DC, rirs, real_noises, real_noise_files)
end
end

function [] = generate_signal(C, corpora_dir, out_dir, rir_dir, sub_set, fs, DC, rirs, real_noises, real_noise_files)
out_file_name   = fullfile(out_dir, sub_set, C{1, 1}{1});
out_file_dir    = fileparts(out_file_name);
speech_file     = fullfile(corpora_dir, 'Speech', C{1, 2}{1});
[speech, fs_in] = audioread(speech_file);
if not(isfolder(out_file_dir))
    mkdir(out_file_dir);
end
if fs_in ~= fs
    speech = resample(speech, fs, fs_in);
end
speech = speech(:);

if strcmp(sub_set, 'test_real')
    x = [conv(speech, rirs{C{1, 9}, C{1, 8}, C{1, 10}}(:, 1)) , conv(speech, rirs{C{1, 9}, C{1, 8}, C{1, 10}}(:, 2))];
else
    airpar = struct('fs', fs, 'rir_type', 1, 'head', 1,...
        'room', C{1, 8}, 'rir_no', C{1, 9}, 'azimuth', C{1, 10});
    airpar.channel  = 1;
    h_left          = load_air(airpar, rir_dir);
    airpar.channel  = 0;
    h_right         = load_air(airpar, rir_dir);
    x               = [conv(speech, h_left(:)) , conv(speech, h_right(:))];
end
out_length      = size(x, 1);

if strcmp(sub_set, 'test_real')
    noise_start    = C{1, 5};
    noise_end      = noise_start + out_length - 1;
    v              = real_noises{find(contains(real_noise_files, C{1, 3}{1}(1:end-1)))}(noise_start:noise_end, :);
else
    noises       = cell(2, 1);
    noise_files  = {fullfile(corpora_dir, 'Noise', C{1, 3}{1}); fullfile(corpora_dir, 'Noise', C{1, 4}{1})};
    noise_starts = [C{1, 5} ; C{1, 6}];
    
    for ch = 1:2
        [noises{ch}, fs_in] = audioread(noise_files{ch});
        if fs_in ~= fs, noises{ch} = resample(noise, fs, fs_in); end
        noises{ch} = noises{ch}(:);
    end
    noise_ends = noise_starts + out_length - 1;
    v = [noises{1}(noise_starts(1):noise_ends(1)), noises{2}(noise_starts(2):noise_ends(2))];
    v = mix_signals(v, DC);
    if size(v,1) < out_length, x = x (1:size(v,1), :); end
end
[~,~,~, gm] = new_v_addnoise(x(:, 1), fs, C{1, 7}, 'dkbSpE', v(:, 1), fs);
y   = x + gm(2) .* v;
y = y./max(max(abs(y)))*(1-(2^-(32-1)));
audiowrite(out_file_name, y, fs, 'BitsPerSample', 32);

% For information, the dbstoi value is tored in C{1, 11}
% it can be computed as:
% val   = dbstoi(x(:, 1), x(:, 2), y(:, 1), y(:, 2), fs);
% difference between val and C{1, 11} should be very small, e.g., < 1e-5
end



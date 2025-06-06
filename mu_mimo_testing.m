clc; clear; close all;

%% Parameters
fc = 2.4e9;
numUsers = 16; numTx = 64; bitsPerUser = 672;
numRB = 52; scs = 30;
carrier = nrCarrierConfig('NSizeGrid', numRB, 'SubcarrierSpacing', scs);
numSubcarriers = 12 * numRB;
numSymbols = 14;
fs = nrOFDMInfo(carrier).SampleRate;
interfPowers = [0 3 6];
numRuns = 20;

% Antenna arrays
d = 0.5 * physconst('LightSpeed') / fc;
txArray = phased.ULA('NumElements', numTx, 'ElementSpacing', d);
rxArray = phased.ULA('NumElements', numUsers, 'ElementSpacing', d);
steer = phased.SteeringVector('SensorArray', txArray, 'PropagationSpeed', physconst('LightSpeed'));

% Beamforming weights
userAngles = linspace(-60, 60, numUsers);
W_fixed = steer(fc, userAngles);
W_fixed = W_fixed ./ vecnorm(W_fixed);

% Subcarrier allocation
scPerUser = floor(numSubcarriers / numUsers);
userSC = arrayfun(@(u) (u-1)*scPerUser + (1:scPerUser), 1:numUsers, 'UniformOutput', false);

% Results
throughput_fixed = zeros(length(interfPowers), numUsers);
latency_fixed = zeros(length(interfPowers), numUsers);
throughput_adaptive = zeros(length(interfPowers), numUsers);
latency_adaptive = zeros(length(interfPowers), numUsers);

for i = 1:length(interfPowers)
    Pintf_dB = interfPowers(i);
    th_fixed = zeros(numRuns, numUsers);
    lat_fixed = zeros(numRuns, numUsers);
    th_adapt = zeros(numRuns, numUsers);
    lat_adapt = zeros(numRuns, numUsers);

    for run = 1:numRuns
        %% Desired Signal
        txWave_desired = zeros(0, numTx);
        userBits = cell(numUsers, 1);
        for u = 1:numUsers
            bits = randi([0 1], bitsPerUser, 1); userBits{u} = bits;
            modSym = nrSymbolModulate(bits, 'QPSK');
            scRange = userSC{u}; numSym = ceil(length(modSym) / length(scRange));
            modSym = [modSym; zeros(length(scRange)*numSym - length(modSym), 1)];
            txGrid = zeros(numSubcarriers, numSymbols);
            txGrid(scRange,1:numSym) = reshape(modSym, length(scRange), numSym);
            waveform = nrOFDMModulate(carrier, txGrid);
            beam = waveform * W_fixed(:,u).';
            if size(txWave_desired,1) < size(beam,1)
                txWave_desired(end+1:size(beam,1), :) = 0;
            end
            txWave_desired = txWave_desired + beam;
        end

        %% Interference Signal
        txWave_intf = zeros(size(txWave_desired));
        for intfIdx = 1:3
            angles = -60 + 120 * rand(1, numUsers);
            W_intf = steer(fc, angles); W_intf = W_intf ./ vecnorm(W_intf);
            waveform_intf = zeros(size(txWave_desired));
            for u = 1:numUsers
                bits = randi([0 1], bitsPerUser, 1);
                modSym = nrSymbolModulate(bits, 'QPSK');
                scRange = userSC{u}; numSym = ceil(length(modSym)/length(scRange));
                modSym = [modSym; zeros(length(scRange)*numSym - length(modSym),1)];
                txGrid = zeros(numSubcarriers, numSymbols);
                txGrid(scRange,1:numSym) = reshape(modSym, length(scRange), numSym);
                waveform = nrOFDMModulate(carrier, txGrid);
                waveform_intf = waveform_intf + waveform * W_intf(:,u).';
            end
            txWave_intf = txWave_intf + 10^(Pintf_dB/20) * waveform_intf;
        end

        %% Channels
        ch_des = nrCDLChannel('DelayProfile','CDL-D','DelaySpread',300e-9,...
            'CarrierFrequency',fc,'SampleRate',fs,...
            'TransmitAntennaArray',txArray,'ReceiveAntennaArray',rxArray);
        ch_intf = clone(ch_des);
        [rx_des, pathGains, st] = ch_des(txWave_desired);
        rx_intf = ch_intf(txWave_intf);
        rx = rx_des + rx_intf;
        rx = awgn(rx, 40, 'measured');
        rxGrid = nrOFDMDemodulate(carrier, rx);

        % Perfect CSI
        pf = getPathFilters(ch_des);
        H_est = nrPerfectChannelEstimate(carrier, pathGains, pf, 0, st);

        % Estimate Rnn from centralized interference
        rx_intf_awgn = awgn(rx_intf, 40, 'measured');
        intfGrid = nrOFDMDemodulate(carrier, rx_intf_awgn);
        samples = reshape(intfGrid, [], numUsers).';
        Rnn = cov(samples');

        %% Adaptive MMSE Combining
        rxGrid_adapt = zeros(numSubcarriers, numSymbols, numUsers);
        for k = 1:numSubcarriers
            for n = 1:numSymbols
                y = squeeze(rxGrid(k,n,:)); % [Nr x 1]
                Hk = squeeze(H_est(k,n,:,:)); % [Nr x Nt]
                H_eff = Hk * W_fixed;         % [Nr x Nu]
                Wrx = (H_eff*H_eff' + Rnn) \ H_eff;
                x_hat = Wrx' * y;
                rxGrid_adapt(k,n,:) = x_hat;
            end
        end

        %% Metrics
        for u = 1:numUsers
            scRange = userSC{u};

            % === Fixed Beamforming Equalization (project channel)
            symbF_raw = reshape(rxGrid(scRange,:,u), [], 1);
            gain = zeros(length(scRange), numSymbols);
            for si = 1:length(scRange)
                k = scRange(si);
                for n = 1:numSymbols
                    Hk = squeeze(H_est(k,n,:,:)); % [Nr x Nt]
                    H_eff = Hk * W_fixed(:,u);   % [Nr x 1]
                    eu = zeros(numUsers,1); eu(u) = 1;
                    gain(si,n) = eu' * H_eff;    % scalar projection
                end
            end
            gain = reshape(gain, [], 1);
            symbF = symbF_raw ./ gain;
            bitsF = nrSymbolDemodulate(symbF, 'QPSK', 'DecisionType','hard');
            bitsF = bitsF(1:bitsPerUser);
            numErrF = sum(bitsF ~= userBits{u});
            th_fixed(run,u) = 100 * (1 - numErrF / bitsPerUser);
            lat_fixed(run,u) = numErrF;

            % === Adaptive
            symbA = reshape(rxGrid_adapt(scRange,:,u), [], 1);
            bitsA = nrSymbolDemodulate(symbA, 'QPSK', 'DecisionType','hard');
            bitsA = bitsA(1:bitsPerUser);
            numErrA = sum(bitsA ~= userBits{u});
            th_adapt(run,u) = 100 * (1 - numErrA / bitsPerUser);
            lat_adapt(run,u) = numErrA;
        end
    end

    throughput_fixed(i,:) = mean(th_fixed);
    latency_fixed(i,:) = mean(lat_fixed);
    throughput_adaptive(i,:) = mean(th_adapt);
    latency_adaptive(i,:) = mean(lat_adapt);
end

%% Plots
figure;
plot(interfPowers, mean(throughput_fixed,2), '-o','LineWidth',2); hold on;
plot(interfPowers, mean(throughput_adaptive,2), '-s','LineWidth',2);
xlabel('Interference Power above Desired (dB)');
ylabel('Average User Throughput (%)');
title('MU-MIMO, 16 UEs, QPSK, CDL-D, fc = 2.4 GHz');
legend('Fixed beamforming','Adaptive beamforming'); grid on;
saveas(gcf, 'throughput_vs_intf.pdf');

figure;
plot(interfPowers, mean(latency_fixed,2), '-o','LineWidth',2); hold on;
plot(interfPowers, mean(latency_adaptive,2), '-s','LineWidth',2);
xlabel('Interference Power above Desired (dB)');
ylabel('Average Bit Errors per User');
title('MU-MIMO, 16 UEs, QPSK, CDL-D, fc = 2.4 GHz');
legend('Fixed beamforming','Adaptive beamforming'); grid on;
saveas(gcf, 'latency_vs_intf.pdf');

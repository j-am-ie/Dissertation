clc; clear; close all;

%% Simulation Parameters
fc = 2.4e9; 
c = physconst('LightSpeed'); 
lambda = c / fc; 
elementSpacing = lambda / 2; 
numElements = 8; 
fs = 1e6; 
t = (0:1/fs:1e-2)'; 
signalAmplitude = 2; 
interferenceAmplitude = 1; 

% Thermal noise power calculation
k = physconst('Boltzmann');
T0 = 290; 
B = fs; 
thermalNoisePower = k * T0 * B;

% User grid setup
xRange = 0:2:20;
yRange = 0:2:20;
[ux, uy] = meshgrid(xRange, yRange);
userPositions = [ux(:), uy(:), zeros(numel(ux),1)];
numUsers = size(userPositions, 1);

% Layout types
layoutTypes = {'hex', 'optimal', 'corridor', 'poor'};
metrics = struct();
numAPs = 10;

for layoutIdx = 1:length(layoutTypes)
    layoutType = layoutTypes{layoutIdx};
    switch layoutType
        case 'hex'
            radius = 8; center = [10, 10];
            angles = linspace(0, 2*pi, numAPs+1); angles(end) = [];
            apPositions = center + radius * [cos(angles(:)), sin(angles(:))];
            apPositions(:,3) = 0;
        case 'optimal'
            % Optimal placement using K-means clustering on user positions
            rng(1);
            [~, apXY] = kmeans(userPositions(:,1:2), numAPs, 'Replicates',5);
            apPositions = [apXY, zeros(numAPs,1)]; % Removed jitter
        case 'corridor'
            apY = linspace(2, 18, 5);
            apPositions = [repmat(0,5,1), apY', zeros(5,1); ...
                           repmat(20,5,1), apY', zeros(5,1)];
        case 'poor'
            rng(2);
            apPositions = [5*rand(numAPs,1), 5*rand(numAPs,1), zeros(numAPs,1)];
    end

    bestSNR = zeros(numUsers, 1);
    bestAP = zeros(numUsers, 1);
    interferencePowerMap = zeros(numUsers, 1);

    for u = 1:numUsers
        userPos = userPositions(u,:);
        maxSNR = -Inf;

        for ap = 1:numAPs
            apPos = apPositions(ap,:);
            relPos = userPos - apPos;
            doa = [atan2d(relPos(2), relPos(1)); 0];

            array = phased.ULA('NumElements', numElements, 'ElementSpacing', elementSpacing);
            collector = phased.Collector('Sensor', array, 'PropagationSpeed', c, 'OperatingFrequency', fc);

            data = randi([0 1], length(t), 1);
            desiredSignal = signalAmplitude * (2*data - 1);
            desiredWave = collector(desiredSignal, doa);

            % Enhanced interference modeling
            distances = vecnorm(apPositions(:,1:2) - userPos(1:2), 2, 2);
            [~, sortedAPs] = sort(distances);
            sortedAPs(sortedAPs == ap) = [];
            numInts = min(3, numel(sortedAPs));
            interferers = sortedAPs(1:numInts);

            interferenceWave = zeros(size(desiredWave));
            for intAP = interferers'
                intRelPos = userPos - apPositions(intAP,:);
                intDoa = [atan2d(intRelPos(2), intRelPos(1)); 0];
                interferenceData = randi([0 1], length(t), 1);
                interferenceSignal = (2*interferenceData - 1);

                % Path loss model (Free-space)
                d = norm(intRelPos);
                if d == 0
                    d = 0.1; % Avoid division by zero
                end
                ploss = (lambda / (4*pi*d))^2;

                % Scale interference amplitude by path loss
                scaledIntSignal = sqrt(interferenceAmplitude^2 * ploss) * interferenceSignal;
                interferenceWave = interferenceWave + collector(scaledIntSignal, intDoa);
            end

            noise = sqrt(thermalNoisePower/2) * (randn(size(desiredWave)) + 1i*randn(size(desiredWave)));
            rxSignal = desiredWave + interferenceWave + noise;

            steeringVector = phased.SteeringVector('SensorArray', array, 'PropagationSpeed', c);
            sv = steeringVector(fc, doa);
            sv = sv / norm(sv);

            % Stabilized covariance matrix
            Rraw = (rxSignal' * rxSignal) / size(rxSignal, 1);
            epsilon = 1e-2 * trace(Rraw) / numElements;
            R = Rraw + epsilon * eye(numElements);

            % Ensure R is Hermitian and positive semidefinite
            R = (R + R') / 2;
            [~, p] = chol(R);
            if p > 0
                warning('Covariance matrix not positive semidefinite. Using fallback MVDR weights.');
                w = R \ sv;
                w = w / (sv' * w);
            else
                try
                    w = lcmvweights(sv, 1, R);
                catch
                    warning('LCMV failed; falling back to MVDR weights.');
                    w = R \ sv;
                    w = w / (sv' * w);
                end
            end

            output = rxSignal * w;

            signalPower = mean(abs(output).^2);
            interferencePower = mean(abs(interferenceWave(:)).^2);
            totalNoisePower = interferencePower + thermalNoisePower;
            snr = 10 * log10(signalPower / totalNoisePower);

            if snr > maxSNR
                maxSNR = snr;
                selectedAP = ap;
                userInterferencePower = interferencePower;
            end
        end

        bestSNR(u) = maxSNR;
        bestAP(u) = selectedAP;
        interferencePowerMap(u) = userInterferencePower;
    end

    ber = berawgn(bestSNR, 'psk', 2, 'nondiff');
    capacity = mean(log2(1 + 10.^(bestSNR/10)));
    coverage = mean(bestSNR > 5);
    fairness = (sum(bestSNR)^2) / (numUsers * sum(bestSNR.^2));

    metrics.(layoutType).snr = bestSNR;
    metrics.(layoutType).ber = mean(ber);
    metrics.(layoutType).capacity = capacity;
    metrics.(layoutType).coverage = coverage;
    metrics.(layoutType).fairness = fairness;
    metrics.(layoutType).interference = interferencePowerMap;

    snrMap = reshape(bestSNR, length(yRange), length(xRange));
    figure; imagesc(xRange, yRange, snrMap); set(gca, 'YDir', 'normal');
    title(sprintf('SNR Map - %s Layout', layoutType)); colorbar;

    intMap = reshape(interferencePowerMap, length(yRange), length(xRange));
    figure; imagesc(xRange, yRange, intMap); set(gca, 'YDir', 'normal');
    title(sprintf('Interference Power Map - %s Layout', layoutType)); colorbar;

    figure; scatter(userPositions(:,1), userPositions(:,2), 30, bestAP, 'filled');
    colormap(hsv(numAPs)); colorbar;
    hold on;
    plot(apPositions(:,1), apPositions(:,2), 'ks', 'MarkerFaceColor', 'w', 'MarkerSize', 10);
    text(apPositions(:,1)+0.3, apPositions(:,2), ...
        arrayfun(@(n) sprintf('AP%d', n), 1:numAPs, 'UniformOutput', false), ...
        'Color', 'k', 'FontSize', 8);
    xlabel('X (m)'); ylabel('Y (m)');
    title(sprintf('User-AP Association Map - %s Layout', layoutType));
    axis equal; grid on;
end

%% Export Metrics
outputFile = 'layout_metrics.xlsx';
layoutNames = layoutTypes(:);
berList = zeros(length(layoutTypes), 1);
capacityList = zeros(length(layoutTypes), 1);
coverageList = zeros(length(layoutTypes), 1);
fairnessList = zeros(length(layoutTypes), 1);

for i = 1:length(layoutTypes)
    layout = layoutTypes{i};
    berList(i) = metrics.(layout).ber;
    capacityList(i) = metrics.(layout).capacity;
    coverageList(i) = metrics.(layout).coverage * 100;
    fairnessList(i) = metrics.(layout).fairness;
end

T = table(layoutNames, berList, capacityList, coverageList, fairnessList, ...
    'VariableNames', {'Layout', 'AvgBER', 'AvgCapacity_bpsHz', 'Coverage_percent', 'FairnessIndex'});
writetable(T, outputFile);
fprintf('Metrics exported to: %s\n', outputFile);

%% Display Results
fprintf('\n=== Layout Comparison Summary ===\n');
for i = 1:length(layoutTypes)
    layout = layoutTypes{i};
    fprintf('Layout: %s\n', layout);
    fprintf('  Avg BER      : %.4e\n', metrics.(layout).ber);
    fprintf('  Avg Capacity : %.2f bps/Hz\n', metrics.(layout).capacity);
    fprintf('  Coverage     : %.2f%%\n', metrics.(layout).coverage * 100);
    fprintf('  Fairness     : %.4f\n\n', metrics.(layout).fairness);
end

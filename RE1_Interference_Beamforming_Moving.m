clear; clc; close all;

%% Parameters
M = 8; d = 0.5; snapshots = 1000;
SNR_dB = 20; SNR = 10^(SNR_dB/10);
K_intf = 6; INR_dB_vec = -10:5:30; runs = 200;
algorithms = {'No Beamforming','LMS','NLMS','RLS','MVDR','SMI'};
num_algorithms = numel(algorithms);
rng(0); % For reproducibility

%% Storage
SINR_all = zeros(num_algorithms, runs, numel(INR_dB_vec));
theta_scan = -90:0.5:90;
beam_patterns = zeros(num_algorithms, numel(theta_scan));

%% INR Sweep
for iINR = 1:numel(INR_dB_vec)
    INR = 10^(INR_dB_vec(iINR)/10);
    noise_power = 1 / SNR;
    interfere_power = 1 / INR;

    for run = 1:runs
        theta_desired = -60 + 120*rand;
        theta_intf = -60 + 120*rand(1, K_intf);
        angles = [theta_desired theta_intf];

        A = exp(1j*2*pi*d*(0:M-1)'*sind(angles));
        s = sqrt(0.5)*(randn(1,snapshots) + 1j*randn(1,snapshots));
        I = sqrt(0.5*interfere_power)*(randn(K_intf,snapshots) + 1j*randn(K_intf,snapshots));
        n = sqrt(0.5*noise_power)*(randn(M,snapshots) + 1j*randn(M,snapshots));
        X = A(:,1)*s + A(:,2:end)*I + n;

        for algIdx = 1:num_algorithms
            switch algorithms{algIdx}
                case 'No Beamforming'
                    w = A(:,1)/M;

                case 'LMS'
                    mu = 0.01; w = zeros(M,1);
                    for k = 1:snapshots
                        x_k = X(:,k);
                        e = s(k) - w'*x_k;
                        w = w + mu * x_k * conj(e);
                    end

                case 'NLMS'
                    mu = 0.01; w = zeros(M,1);
                    for k = 1:snapshots
                        x_k = X(:,k);
                        e = s(k) - w'*x_k;
                        w = w + (mu/(norm(x_k)^2 + 1e-6)) * x_k * conj(e);
                    end

                case 'RLS'
                    lambda = 0.99; delta = 0.01;
                    P = eye(M)/delta; w = zeros(M,1);
                    for k = 1:snapshots
                        x_k = X(:,k);
                        pi_k = P*x_k;
                        k_k = pi_k / (lambda + x_k'*pi_k);
                        e = s(k) - w'*x_k;
                        w = w + k_k * conj(e);
                        P = (P - k_k*x_k'*P)/lambda;
                    end

                case 'MVDR'
                    R = (X * X') / snapshots;
                    epsilon = 1e-1;
                    R = R + eye(M) * epsilon;
                    if cond(R) > 1e6
                        SINR_all(algIdx, run, iINR) = NaN;
                        continue;
                    end
                    a_d = A(:,1);
                    w_mvdr = R \ a_d;
                    denom = a_d' * w_mvdr;
                    if abs(denom) < 1e-6
                        SINR_all(algIdx, run, iINR) = NaN;
                        continue;
                    end
                    w = w_mvdr / denom;

                case 'SMI'
                    R = (X * X') / snapshots;
                    R = R + eye(M) * 1e-3;
                    w = R \ A(:,1);
                    w = w / (A(:,1)' * w);
            end

            w = w / norm(w);
            y = w' * X;
            desired = w' * A(:,1) * s;
            residual = y - desired;
            signal_power = mean(abs(desired).^2);
            interference_power = mean(abs(residual).^2);
            if isfinite(signal_power) && isfinite(interference_power) && interference_power > 0
                SINR_all(algIdx, run, iINR) = signal_power / interference_power;
            else
                SINR_all(algIdx, run, iINR) = NaN;
            end
        end
    end
end

%% Performance Metrics
SINR_mean = 10*log10(nanmean(SINR_all,2));
SINR_p5 = 10*log10(prctile(SINR_all,5,2));
Pout = mean(SINR_all < 10^(10/10),2);

%% Visualization

% SINR vs INR
figure;
hold on;
for algIdx = 1:num_algorithms
    sinr_vals = squeeze(SINR_mean(algIdx,1,:));
    if all(isfinite(sinr_vals))
        plot(INR_dB_vec, sinr_vals, '-o', 'DisplayName', algorithms{algIdx});
    end
end
xlabel('INR (dB)');
ylabel('Average Output SINR (dB)');
title('SINR vs INR for Various Beamforming Algorithms');
legend('Location','best'); grid on;

% CDF at highest INR
figure;
hold on;
for algIdx = 1:num_algorithms
    SINR_data = squeeze(SINR_all(algIdx,:,end));
    SINR_data = SINR_data(isfinite(SINR_data));
    if ~isempty(SINR_data)
        [f, x] = ecdf(10*log10(SINR_data));
        plot(x, f, 'DisplayName', algorithms{algIdx});
    end
end
xlabel('Output SINR (dB)');
ylabel('Cumulative Probability');
title('CDF of Output SINR at INR = 30 dB');
legend('Location','best'); grid on;

% Beam Patterns
for algIdx = 1:num_algorithms
    w_sum = zeros(M,1);
    for run = 1:runs
        theta_desired = -60 + 120*rand;
        theta_interfere = -60 + 120*rand(1,K_intf);
        angles = [theta_desired theta_interfere];
        A = exp(1j*2*pi*d*(0:M-1)'*sind(angles));
        R = (A(:,2:end)*A(:,2:end)') / K_intf + eye(M) * noise_power;
        a_d = A(:,1);
        w = (R \ a_d) / (a_d' * (R \ a_d));
        w_sum = w_sum + w;
    end
    w_avg = w_sum / runs;
    for idx = 1:length(theta_scan)
        sv = exp(1j*2*pi*d*(0:M-1)'*sind(theta_scan(idx)));
        beam_patterns(algIdx,idx) = abs(w_avg' * sv);
    end
    beam_patterns(algIdx,:) = 20*log10(beam_patterns(algIdx,:) / max(beam_patterns(algIdx,:)));
end

figure;
hold on;
for algIdx = 1:num_algorithms
    plot(theta_scan, beam_patterns(algIdx,:), 'DisplayName', algorithms{algIdx});
end
xlabel('Angle (degrees)');
ylabel('Normalized Beam Pattern (dB)');
title('Average Beam Patterns Across Runs');
legend('Location','best'); grid on;
ylim([-50 0]);

% Export
saveas(figure(1), 'sinr_vs_inr.pdf');
saveas(figure(2), 'sinr_cdf_30dB.pdf');
saveas(figure(3), 'beam_patterns.pdf');

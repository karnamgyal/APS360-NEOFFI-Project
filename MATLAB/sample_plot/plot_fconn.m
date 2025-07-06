load("fconn_rest_hcp_yeo17network_aseg_mc_1.mat");
for i = 1:17
    figure(i);
    
    % Subplot 1: Correlation coefficient matrix
    subplot(1,2,1);
    imagesc(fconn{1}.fconn{1,i+1}.fconn_corrcoef);    
    colormap(hsv);
    colorbar;
    title(['Corrcoef for index ', num2str(i+1)]);
    
    % Subplot 2: Dimension reduction
    subplot(1,2,2);
    imagesc(fconn{1}.fconn{1,i+1}.fconn_u10 * diag(fconn{1}.fconn{1,i+1}.fconn_s10) * fconn{1}.fconn{1,i+1}.fconn_u10');
    colormap(hsv);
    colorbar;
    title(['Reduced (10D) for index ', num2str(i+1)]);
    
    pause(1);
end

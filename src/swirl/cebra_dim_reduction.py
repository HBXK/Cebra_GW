import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from cebra import CEBRA
import matplotlib
matplotlib.use("Agg")
import plotly.graph_objs as go
from plotly.offline import plot


# =============== #
# CLASS: CEBRAUtils
# =============== #

class CEBRAUtils:
    """
    Static utility methods for applying CEBRA, computing metrics, plotting, etc.
    These were originally from 'CEBRA_Utils.py'.
    """

    @staticmethod
    def apply_cebra(neural_data=None, output_dimension=3, temperature=1):
        """
        Apply the CEBRA model to 'neural_data' and return the embeddings.
        """
        model = CEBRA(
            output_dimension=output_dimension,
            max_iterations=1000,
            batch_size=512,
            temperature=temperature
        )
        model.fit(neural_data)
        embeddings = model.transform(neural_data)
        return embeddings
   
    @staticmethod
    def linear_interpolate_nans_2d(array_2d):
        """
        Replace NaNs with linearly interpolated values (per column).
        array_2d has shape (N, dim).
        """
        x = np.arange(len(array_2d))
        for dim in range(array_2d.shape[1]):
            col = array_2d[:, dim]
            # Identify where col is non-NaN
            good = ~np.isnan(col)
            if np.all(~good):
                # If the entire column is NaN, we can't interpolate.
                continue
            # Interpolate over the NaNs (including boundaries)
            array_2d[~good, dim] = np.interp(x[~good], x[good], col[good])
        return array_2d
   
    @staticmethod
    def linear_interpolate_nans_1d(array_1d):
        """
        Replaces NaN entries in a 1D NumPy array by linear interpolation.
        """
        x = np.arange(len(array_1d))
        good = ~np.isnan(array_1d)
        if np.all(~good):
            # If the entire array is NaN, do nothing or fill as needed
            return array_1d
        array_1d[~good] = np.interp(x[~good], x[good], array_1d[good])
        return array_1d
       



    @staticmethod
    def derivative_and_mv_avg(data=None, window_size=3):
        """
        Compute finite differences and then the moving average
        of those differences, essentially a smoothing operation.
        """
        diffs = np.diff(data)
        kernel = np.ones(window_size) / window_size
        avg_diffs = np.convolve(diffs, kernel, mode='valid')
        return avg_diffs

    @staticmethod
    def nt_TDA_mask(data, pct_distance=1, pct_neighbors=50, pct_dist=80, verbose=False):
        """
        Detect outliers in 'data' using the TDA-inspired method,
        **but return a boolean mask** of the same length as 'data',
        indicating which rows are inliers (True) vs. outliers (False).
        """
        from scipy.spatial import distance_matrix
        from sklearn.neighbors import NearestNeighbors

        # data.shape = (N, dim)
        N = data.shape[0]
        if(verbose == True):
            print(f"[DEBUG] Running nt_TDA_mask on data of shape {data.shape}")
            print(f"[DEBUG] Using params: pct_distance={pct_distance}, pct_neighbors={pct_neighbors}, pct_dist={pct_dist}")


        distances = distance_matrix(data, data)
        if(verbose == True):
            print(f"[DEBUG] distance_matrix shape: {distances.shape} (should be (N, N))")

        # Fill diagonal so points don't count themselves
        np.fill_diagonal(distances, 10)

        # Neighborhood radius for each point
        neighborhood_radius = np.percentile(distances, pct_distance, axis=0)
        if(verbose == True):
            print(f"[DEBUG] neighborhood_radius: min={neighborhood_radius.min():.3f}, "
            f"max={neighborhood_radius.max():.3f}, median={np.median(neighborhood_radius):.3f}")

        neighbor_counts = np.sum(distances <= neighborhood_radius[:, None], axis=1)
        threshold_neighbors = np.percentile(neighbor_counts, pct_neighbors)
        if(verbose == True):
            print(f"[DEBUG] threshold_neighbors (pct_neighbors={pct_neighbors}) is {threshold_neighbors:.3f}")

        outlier_indices_1 = np.where(neighbor_counts < threshold_neighbors)[0]
        if(verbose == True):
            print(f"[DEBUG] outlier_indices_1 has length {len(outlier_indices_1)}")
        # "Far out" outliers using mean min-distance
        neighbgraph = NearestNeighbors(n_neighbors=5).fit(distances)
        dists, _ = neighbgraph.kneighbors(distances)
        min_distance_to_any_point = np.mean(dists, axis=1)
        if(verbose == True):
            print(f"[DEBUG] min_distance_to_any_point: min={min_distance_to_any_point.min():.3f}, "
            f"max={min_distance_to_any_point.max():.3f}, median={np.median(min_distance_to_any_point):.3f}")

        distance_threshold = np.percentile(min_distance_to_any_point, pct_dist)
        if(verbose == True):
            print(f"[DEBUG] distance_threshold (pct_dist={pct_dist}) is {distance_threshold:.3f}")

        outlier_indices_2 = np.where(min_distance_to_any_point > distance_threshold)[0]
        if(verbose == True):
            print(f"[DEBUG] outlier_indices_2 has length {len(outlier_indices_2)}")

        all_outliers = np.unique(np.concatenate([outlier_indices_1, outlier_indices_2]))
        if(verbose == True):
            print(f"[DEBUG] Total unique outliers: {len(all_outliers)}")
        # Construct boolean mask: True = inlier, False = outlier

       
   
        mask = np.ones(N, dtype=bool)
        mask[all_outliers] = False
        inlier_count = np.sum(mask)
        if(verbose == True):
            print(f"[DEBUG] Number of inliers: {inlier_count}, out of N={N}")
            print(f"[DEBUG] Outlier ratio = {1.0 - inlier_count / N:.3f}")
       


        return mask






    @staticmethod
    def low_pass_filter(angles=None, cutoff_frequency=0.1, filter_order=3, fs=1):
        """
        Apply a low-pass Butterworth filter to smooth the passed array of angles.
        """
        from scipy.signal import butter, filtfilt

        nyquist = 0.5 * fs
        normalized_cutoff = cutoff_frequency / nyquist
        b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
        smoothed_angles = filtfilt(b, a, angles)
        return smoothed_angles

    @staticmethod
    def compute_moving_average(data=None, window_size=None):
        """
        Computes the moving average of a 1D array using a simple rectangular kernel.
        """
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='valid')

    @staticmethod
    def get_var_over_lap(var=None, true_angle=None):
        """
        Computes the lap number for each item in 'var' based on 'true_angle'
        and returns the paired and sorted arrays.
        """
        min_length = min(len(var), len(true_angle))
        var = var[:min_length]
        true_angle = true_angle[:min_length]
        lap_number = true_angle / (2 * np.pi)
        sorted_indices = np.argsort(lap_number)
        sorted_lap_number = lap_number[sorted_indices]
        sorted_var = var[sorted_indices]
        return lap_number, sorted_var, sorted_lap_number

    @staticmethod
    def save_data_to_csv(data_dict, save_dir, is_list=False):
        """
        Saves the provided data dictionary or list of dictionaries into a CSV file.
        """
        import pandas as pd
        os.makedirs(save_dir, exist_ok=True)
        if is_list:
            df = pd.DataFrame(data_dict)
            csv_file = os.path.join(save_dir, "total_results.csv")
        else:
            df = pd.DataFrame([data_dict])
            csv_file = os.path.join(save_dir, "results.csv")
        df = df.round(2)
        df.to_csv(csv_file, index=False)


    @staticmethod
    def compute_moving_average(data=None, window_size=None):
        """
        Computes the moving average of a 1D NumPy array.

        Parameters:
        - data (np.ndarray): Input data array.
        - window_size (int): Size of the moving window.

        Returns:
        - np.ndarray: Moving averaged data.
        """
       
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='valid')




# =============== #
# CLASS: CEBRAAnalysis
# =============== #

class CEBRAAnalysis:
    """
    Encapsulates the main analysis steps,
    using the utility methods from CEBRAUtils.
    """

    def __init__(self, data_path, session_choose=True, max_num_reruns=1):
        """
        Loads data, sets up configuration, etc.
        """
        import scipy.io

        # --- Load data ---
        path_flow = os.path.join(
            data_path
        )
        data_flow = scipy.io.loadmat(
            path_flow,
            squeeze_me=True,
            struct_as_record=False
        )
        self.expt_optic_flow = data_flow['expt']



        print(f"Optic Flow expt shape: {self.expt_optic_flow.shape}")
        # print(f"Landmark expt shape: {self.expt_landmark.shape}")

        # --- Config parameters ---
        self.session_choose = session_choose
        if self.session_choose:
            # self.landmark_sessions = []
            self.optic_flow_sessions = []
        else:
            # self.landmark_num_trials = 65
            # self.landmark_control_point = 42
            self.optic_flow_num_trials = 72
            self.optic_flow_control_point = 40


        self.max_num_reruns = max_num_reruns
        self.save_folder = 'betti'

        # List of experiment definitions
        self.expts = [
            ("optic_flow", self.expt_optic_flow,
             getattr(self, 'optic_flow_control_point', None),
             getattr(self, 'optic_flow_num_trials', None)),

            # ("landmark", self.expt_landmark,
            #  getattr(self, 'landmark_control_point', None),
            #  getattr(self, 'landmark_num_trials', None))
        ]

        # More config
        # self.model_save_path = os.path.join('/Users/devenshidfar/Desktop/Masters/NRSC_510B/',
        #                                     'cebra_control_recal/models')
        self.save_models = 1
        self.save_anim = 1
        self.load_npy = 0
        self.rm_outliers = True
        self.vel_threshold = 5  # degrees per second
        self.bin_sizes = [1]
        self.max_num_reruns = max_num_reruns

        # For final data collection
        self.all_neural_data = []
        self.all_embeddings_3d = []
        self.H0_value = []
        self.H1_value = []
        self.all_betti_0 = []
        self.all_betti_1 = []
        self.all_principal_curves_3d = []
        self.all_curve_params_3d = []
        self.all_binned_hipp_angle = []
        self.all_binned_true_angle = []
        self.all_binned_est_gain = []
        self.all_binned_high_vel = []
        self.all_decoded_angles = []
        self.all_filtered_decoded_angles_unwrap = []
        self.all_decode_H = []
        self.all_session_idx = []
        self.all_rat = []
        self.all_day = []
        self.all_epoch = []
        self.all_num_skipped_clusters = []
        self.all_num_used_clusters = []
        self.all_avg_skipped_cluster_isolation_quality = []
        self.all_avg_used_cluster_isolation_quality = []
        self.all_mean_distance_to_principal_curve = []
        self.all_mean_angle_difference = []
        self.all_shuffled_mean_angle_difference = []
        self.all_SI_score_hipp = []
        self.all_SI_score_true = []
        # self.all_mse_decode_vs_true = []
        self.all_mean_H_difference = []
        self.all_std_H_difference = []

    def run_analysis(self, embedding_folder_path):
        """
        Main analysis pipeline.
        Loops over expts (optic_flow, landmark),
        loads sessions, runs embeddings, decodes, etc.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats
        from matplotlib.backends.backend_pdf import PdfPages
        from scipy.signal import savgol_filter
        from scipy.io import savemat


        for expt_name, expt, control_point, num_trials in self.expts:
            control_count = 0
            print(f"Control point: {control_point} and num_trials: {num_trials}")
            if control_point is None or num_trials is None:
                # If session_choose=True, skip the logic of skip/stop
                control_point = 0
                num_trials = len(expt)

            # For each bin size
            for bin_size in self.bin_sizes:
                print(f"\n[INFO] Processing expt: {expt_name}, bin_size: {bin_size}s")
                a = 0
                for session_idx, session in enumerate(expt):
                    try:
                        control_count += 1
                        # Skip control sessions
                        if control_count <= control_point:
                            print(f"Skipping session {session_idx + 1} (control count <= control_point).")
                            continue
                        elif control_count > (control_point + num_trials):
                            print("[INFO] Reached desired number of trials. Exiting session loop.")
                            break

                        print(f"\n[INFO] Processing session {session_idx}/{len(expt)} in {expt_name} experiments")
                        print(f"Rat: {session.rat}, Day: {session.day}, Epoch: {session.epoch}")

                        session_base_path = os.path.join(
                        'C:\\Users\\zziyu\\Desktop\\CEBRA\\.venv\\Data\\Deven_Data',
                        self.save_folder,
                        expt_name,
                        f'rat_{session.rat}',
                        f'session_{session_idx}'
                        )  
                        os.makedirs(session_base_path, exist_ok=True)

                        SI_plots_path = os.path.join(session_base_path, 'SI_Plots')
                        anim_save_file = os.path.join(session_base_path, '3d_Animations')
                        spectrogram_path = os.path.join(session_base_path, 'Spatial_Spectrograms')
                        param_plot_path = os.path.join(session_base_path, 'Param_Plots')
                        H_plot_path = os.path.join(session_base_path, 'H_Plots')
                        pers_hom_path = os.path.join(session_base_path, 'Pers_Hom_Plots')
                       

                        paths_to_create = [SI_plots_path, anim_save_file, spectrogram_path, param_plot_path, H_plot_path]
                        for path in paths_to_create:
                            os.makedirs(path, exist_ok=True)

                        ros_data = session.rosdata
                        start_time = ros_data.startTs
                        end_time = ros_data.stopTs

                        enc_times = np.array(ros_data.encTimes - start_time) / 1e6
                        vel = np.array(ros_data.vel)
                        valid_idx = np.isfinite(enc_times) & np.isfinite(vel)
                        enc_times = enc_times[valid_idx]
                        vel = vel[valid_idx]
                        high_vel_idx = vel > self.vel_threshold
                        if np.sum(high_vel_idx) == 0:
                            print("[WARNING] No data points above velocity threshold. Skipping session.")
                            continue

                        enc_times_high_vel = enc_times[high_vel_idx]
                        high_vel_filtered = vel[high_vel_idx]
                        est_gain_filtered = np.array(ros_data.estGain)[valid_idx][high_vel_idx]
                        hipp_angle_filtered = np.array(ros_data.hippAngle)[valid_idx][high_vel_idx]
                        true_angle_filtered = np.array(ros_data.encAngle)[valid_idx][high_vel_idx]
                        rel_angle_filtered = np.array(ros_data.relAngle)[valid_idx][high_vel_idx]

                        bins = np.arange(enc_times_high_vel[0], enc_times_high_vel[-1] + bin_size, bin_size)
                        if len(bins) < 2:
                            print("[WARNING] Not enough bins after filtering for high velocity. Skipping session.")
                            continue
                        try:
                            binned_est_gain, _, _ = stats.binned_statistic(
                                enc_times_high_vel, est_gain_filtered, statistic='mean', bins=bins
                            )
                            binned_hipp_angle, _, _ = stats.binned_statistic(
                                enc_times_high_vel, hipp_angle_filtered, statistic='mean', bins=bins
                            )
                            binned_true_angle, _, _ = stats.binned_statistic(
                                enc_times_high_vel, true_angle_filtered, statistic='mean', bins=bins
                            )
                            binned_high_vel, _, _ = stats.binned_statistic(
                                enc_times_high_vel, high_vel_filtered, statistic='mean', bins=bins
                            )
                            binned_rel_angle, _, _ = stats.binned_statistic(
                                enc_times_high_vel, rel_angle_filtered, statistic='mean', bins=bins
                            )

           
                        except ValueError as e:
                            # Catch "Bin edges must be unique"
                            print(f"[WARNING] Binning failed for session {session_idx}. Exception: {e}")
                            # Skip this session and continue with the next
                            continue

                        valid_bins = (
                            ~np.isnan(binned_hipp_angle) &
                            ~np.isnan(binned_true_angle) &
                            ~np.isnan(binned_est_gain) &
                            ~np.isnan(binned_high_vel)
                        )
                        print("valid bins")
                        print(valid_bins)


                        if not np.all(valid_bins):
                            binned_hipp_angle = binned_hipp_angle[valid_bins]
                            binned_true_angle = binned_true_angle[valid_bins]
                            binned_est_gain = binned_est_gain[valid_bins]
                            binned_high_vel = binned_high_vel[valid_bins]
                            binned_rel_angle = binned_rel_angle[valid_bins]
                            bins = bins[:-1][valid_bins]



                        # Filter spike times
                        all_spikes = []
                        skipped_clusters = 0
                        used_clusters = 0
                        used_cluster_iq_list = []
                        skipped_cluster_iq_list = []

                        for cluster in session.clust:
                            if cluster.isolationQuality > 4:
                                skipped_clusters += 1
                                skipped_cluster_iq_list.append(cluster.isolationQuality)
                                continue
                            else:
                                used_clusters += 1
                                used_cluster_iq_list.append(cluster.isolationQuality)

                            spike_times_sec = (cluster.ts - start_time) / 1e6
                            vel_at_spikes = cluster.vel
                            include_spikes = vel_at_spikes > self.vel_threshold
                            spike_times_sec_high_vel = spike_times_sec[include_spikes]
                            if len(spike_times_sec_high_vel) == 0:
                                continue
                           
                            try:
                                binned_spikes, _, _ = stats.binned_statistic(
                                    spike_times_sec_high_vel,
                                    np.ones_like(spike_times_sec_high_vel),
                                    statistic='sum',
                                    bins=bins
                                )
                                all_spikes.append(binned_spikes)
                            except ValueError as e:
                                # Catch "Bin edges must be unique"
                                print(f"[WARNING] Binning failed for session {session_idx}. Exception: {e}")
                                # Skip this session and continue with the next
                                continue
                       
       
                        # Stats for cluster
                        used_cluster_iq = np.asarray(used_cluster_iq_list)
                        skipped_cluster_iq = np.asarray(skipped_cluster_iq_list)
                        num_skipped_cluster = len(skipped_cluster_iq)
                        num_used_cluster = len(used_cluster_iq)
                        avg_skipped_cluster_iq = np.mean(skipped_cluster_iq) if len(skipped_cluster_iq) else np.nan
                        avg_used_cluster_iq = np.mean(used_cluster_iq) if len(used_cluster_iq) else np.nan

                        if not all_spikes:
                            print("[WARNING] No valid spike data after filtering. Skipping session.")
                            continue

                        # Build neural data
                        neural_data = np.array(all_spikes).T
                        num_bins_neural = neural_data.shape[0]
                        num_bins_behavior = len(binned_est_gain)
                        if num_bins_neural != num_bins_behavior:
                            min_bins = min(num_bins_neural, num_bins_behavior)
                            neural_data = neural_data[:min_bins, :]
                            binned_est_gain = binned_est_gain[:min_bins]
                            binned_hipp_angle = binned_hipp_angle[:min_bins]
                            binned_true_angle = binned_true_angle[:min_bins]
                            binned_high_vel = binned_high_vel[:min_bins]
                            bins = bins[:min_bins]

                        # For reruns if SI < threshold
                        # pdf_filename = os.path.join(session_base_path, f'session_{session_idx}.pdf')
                        # os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)
                        # pdf = PdfPages(pdf_filename)

                        temperature_list = [1]
                        best_embeddings_3d = None
                        best_SI_score_hipp = -999

                        # Attempt multiple runs if SI < some threshold
                        skip_session = False
                        for temp in temperature_list:
                            rerun_count = 0
                            while rerun_count < self.max_num_reruns:
                                embeddings_high_dim = CEBRAUtils.apply_cebra(
                                    neural_data=neural_data,
                                    output_dimension=3,
                                    temperature=temp
                                )
                                embeddings_3d = embeddings_high_dim.copy()


                                # Outlier removal
                                if self.rm_outliers:
                                    # Get full-length boolean mask
                                    inlier_mask_3d = CEBRAUtils.nt_TDA_mask(embeddings_3d,verbose=True)
                                    outlier_mask_3d = ~inlier_mask_3d

                                    # Set outliers to NaN
                                    embeddings_3d[outlier_mask_3d, :] = np.nan

                                    # The binned_x arrays correspond 1-to-1 with embeddings_3d
                                    binned_hipp_angle[outlier_mask_3d] = np.nan
                                    binned_true_angle[outlier_mask_3d] = np.nan
                                    binned_est_gain[outlier_mask_3d] = np.nan
                                    binned_high_vel[outlier_mask_3d] = np.nan

                                    # Interpolate so there are no NaNs but the same shape
                                    embeddings_3d = CEBRAUtils.linear_interpolate_nans_2d(embeddings_3d)
                                    binned_hipp_angle = CEBRAUtils.linear_interpolate_nans_1d(binned_hipp_angle)
                                    binned_true_angle = CEBRAUtils.linear_interpolate_nans_1d(binned_true_angle)
                                    binned_est_gain = CEBRAUtils.linear_interpolate_nans_1d(binned_est_gain)
                                    binned_high_vel = CEBRAUtils.linear_interpolate_nans_1d(binned_high_vel)

                                if embeddings_3d.shape[0] < 500:
                                    print(f"length of embeddings is: {embeddings_3d.shape[0]}, skipping session {session_idx}")
                                    skip_session = True
                                    break




                                # Convert angles to radians
                                binned_true_angle_rad = np.deg2rad(binned_true_angle)
                                binned_hipp_angle_rad = np.deg2rad(binned_hipp_angle)



                                binned_true_angle_rad = (binned_true_angle_rad
                                                        % (2 * np.pi))
                                binned_hipp_angle_rad = (binned_hipp_angle_rad
                                                        % (2 * np.pi))
                               
                                embeddings_3d_mean = np.mean(embeddings_3d, axis=0)

                                embeddings_3d = embeddings_3d - embeddings_3d_mean
                                rerun_count += 1


                            

                            if skip_session:
                                print("[INFO] not enough embedding points "
                                        "Skipping the entire session and writing NaNs.")

                                self.all_neural_data.append(np.nan)
                                self.all_embeddings_3d.append(np.nan)
                                self.H0_value.append(np.nan)
                                self.H1_value.append(np.nan)
                                self.all_betti_0.append(np.nan)
                                self.all_betti_1.append(np.nan)
                                self.all_principal_curves_3d.append(np.nan)
                                self.all_curve_params_3d.append(np.nan)
                                self.all_binned_hipp_angle.append(np.nan)
                                self.all_binned_true_angle.append(np.nan)
                                self.all_binned_est_gain.append(np.nan)
                                self.all_binned_high_vel.append(np.nan)
                                self.all_decoded_angles.append(np.nan)
                                self.all_filtered_decoded_angles_unwrap.append(np.nan)
                                self.all_decode_H.append(np.nan)
                                self.all_session_idx.append(session_idx)
                                self.all_rat.append(session.rat)
                                self.all_day.append(session.day)
                                self.all_epoch.append(session.epoch)
                                self.all_num_skipped_clusters.append(np.nan)
                                self.all_num_used_clusters.append(np.nan)
                                self.all_avg_skipped_cluster_isolation_quality.append(np.nan)
                                self.all_avg_used_cluster_isolation_quality.append(np.nan)
                                self.all_mean_distance_to_principal_curve.append(np.nan)
                                self.all_mean_angle_difference.append(np.nan)
                                self.all_shuffled_mean_angle_difference.append(np.nan)
                                self.all_SI_score_hipp.append(np.nan)
                                self.all_SI_score_true.append(np.nan)
                                # self.all_mse_decode_vs_true.append(np.nan)
                                self.all_mean_H_difference.append(np.nan)
                                self.all_std_H_difference.append(np.nan)

                                # Continue to the next session if not enough embedding points
                                skip_session = False
                                continue

                            if best_embeddings_3d is not None:
                                embeddings_3d = best_embeddings_3d
                            else:
                                # fallback
                                embeddings_3d = embeddings_high_dim

                        np.save(f'{embedding_folder_path}\\embedding_{a}.npy', embeddings_3d)
                        a+=1
                        print('SAVED')
                            
                    except:
                        print('error')
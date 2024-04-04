### IN PROGRESS ####
# 
# from concurrent.futures import ProcessPoolExecutor
import numpy as np, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric, MaximumMetric 
from pyrqa.analysis_type import Classic

from pyunicorn.timeseries import RecurrencePlot
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

def embed_time_series(ts, tau, dim):
    """
    Embed a time series with given dimension and time delay.
    
    :param ts: 1D NumPy array or list containing the time series.
    :param dim: The embedding dimension.
    :param tau: The time delay (lag).
    :return: 2D NumPy array containing the embedded time series.
    """
    n = len(ts)
    if n < (dim - 1) * tau + 1:
        raise ValueError("Time series is too short for the given dimension and time delay")

    embedded_ts = np.empty((n - (dim - 1) * tau, dim))
    for i in range(dim):
        embedded_ts[:, i] = ts[i * tau: n - (dim - 1) * tau + i * tau]
    return embedded_ts

def get_rqa_measures(result):
    """
    Extract RQA measures from an RQAResult object and store them in a dictionary.
    
    Parameters:
        result: RQAResult object from pyrqa computation.
    
    Returns:
        dict: Dictionary of RQA measures.
    """
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2

    rqa_measures = {
        'RR': result.recurrence_rate,
        'DET': result.determinism,
        'ADL': result.average_diagonal_line,
        'LDL': result.longest_diagonal_line,
        'DIV': result.divergence,
        'ENTR': result.entropy_diagonal_lines,
        'LAM': result.laminarity,
        'TT': result.trapping_time,
        'LVL': result.longest_vertical_line,
        'VENTR': result.entropy_vertical_lines,
        'AVL': result.average_white_vertical_line,
        'LWVL': result.longest_white_vertical_line,
        'WVLENTR': result.entropy_white_vertical_lines,
        'DET/RR': result.ratio_determinism_recurrence_rate,
        'LAM/DET': result.ratio_laminarity_determinism
    }
    return rqa_measures


def find_epsilon(time_series_data, target_recurrence_rate, tau=1, dim=1, theiler=1, initial_epsilon=1.0, tolerance=0.0001, max_iterations=100):
    """
    Find the epsilon value that corresponds to a given recurrence rate.
    
    Parameters:
        time_series_data (np.array): The time series data.
        target_recurrence_rate (float): The target recurrence rate.
        initial_epsilon (float): Initial guess for epsilon.
        tolerance (float): Tolerance for the difference between current and target recurrence rates.
        max_iterations (int): Maximum number of iterations to perform.
        
    Returns:
        float: The epsilon value corresponding to the target recurrence rate.
    """
    lower_epsilon = 0
    upper_epsilon = initial_epsilon
    current_epsilon = initial_epsilon
    time_series = TimeSeries(time_series_data, embedding_dimension=dim, time_delay=tau)
    
    for _ in range(max_iterations):
        settings = Settings(time_series,
                            analysis_type=Classic,
                            neighbourhood=FixedRadius(current_epsilon),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=tau*dim)
        
        rqa_computation = RQAComputation.create(settings)
        rqa_result = rqa_computation.run()
        current_recurrence_rate = rqa_result.recurrence_rate
        
        if abs(current_recurrence_rate - target_recurrence_rate) < tolerance:
            return current_epsilon  # Found the epsilon corresponding to the target recurrence rate

        # Update the search bounds and current epsilon based on the comparison
        if current_recurrence_rate > target_recurrence_rate:
            upper_epsilon = current_epsilon
            current_epsilon = (lower_epsilon + upper_epsilon) / 2
        else:
            lower_epsilon = current_epsilon
            current_epsilon = (lower_epsilon + upper_epsilon) / 2
    
    return current_epsilon  # Return the best approximation of epsilon after max_iterations

def calculate_rqa_measures(ts, tau=1, dim=1, epsilon=0.05, metric='euclidean', window=-1, step_size=None):
    start = time.time()
    
    if window != -1: # ts is a windowed segment of embedded time series
        time_series = EmbeddedSeries(ts) 

    else: # construct regular time series object, ts is the full time series
        time_series = TimeSeries(ts, time_delay=tau, embedding_dimension=dim)
    
    distance_metrics = {'euclidean': EuclideanMetric, 'supremum': MaximumMetric}
    
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(epsilon),
                        similarity_measure=distance_metrics[metric.lower()],
                        theiler_corrector=tau*dim)
    
    # Perform RQA computation
    result = RQAComputation.create(settings).run()
    
    rqa = get_rqa_measures(result)
    if window != -1:
        rqa['window'] = window  # Add window index to the results
    if step_size: rqa['time'] = window_size + window * step_size # center time of the window

    print(f"window {i} took {time.time() - start} seconds")
    return rqa

def compute_measures_in_parallel(ts, tau, dim, epsilon, metric, window_size, step_size):
    # Compute RQA measures for a single surrogate for all its windows
    # This function will return a DataFrame with results for all windows of one surrogate

    # calculate windows from embedded time series if sliding windowing params supplied
    results = []
    if window_size and step_size: 
        # embed full time series and calculate windows
        embedded_ts = embed_time_series(ts, tau, dim)
        for i in range(0, len(embedded_ts) - window_size + 1, step_size):
            window = embedded_ts[i:i + window_size]
            # Calculate RQA measures for each window
            rqa_result = calculate_rqa_measures(window, tau, dim, epsilon, metric, i, step_size)
            results.append(rqa_result)
    else:
        # Calculate RQA measures for the full time series (no windowing)
        rqa_result = calculate_rqa_measures(ts, tau, dim, epsilon, metric)
        results.append(rqa_result)

    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(results)

from threading import Thread
import queue

from pyrqa.exceptions import SubMatrixNotProcessedException
from pyrqa.opencl import OpenCL
from pyrqa.processing_order import Diagonal
from pyrqa.scalable_recurrence_analysis import SubMatrix, SubMatrices, Verbose
from pyrqa.runtimes import MatrixRuntimes, \
    FlavourRuntimesMonolithic
from pyrqa.selector import SingleSelector

# Make sure to import the necessary classes from the pyrqa package
# from pyrqa.settings import Settings
# from pyrqa.time_series import TimeSeries

class WRQAEngine(BaseEngine):
    def __init__(self, settings, verbose=False, window_size=1024, step_size=1, opencl=None, use_profiling_events_time=True):
        # Initialize the base engine
        super().__init__(settings, verbose, window_size, Diagonal, opencl, use_profiling_events_time)
        # Initialize the new attributes
        self.window_size = window_size
        self.step_size = step_size
        # Override sub_matrix_queues to customize queue creation
        self.sub_matrix_queues = [queue.Queue() for _ in range(self.number_of_partitions_x * self.number_of_partitions_y)]
        self.create_windowed_sub_matrices()

    def create_windowed_sub_matrices(self):
        """
        Create sub matrices based on the windowing approach.
        """
        self.number_of_partitions_x = (len(self.settings.time_series_y.data) - self.window_size) // self.step_size + 1
        self.number_of_partitions_y = self.number_of_partitions_x  # Assuming square sub-matrices for simplicity
        self.sub_matrix_queues = [queue.Queue() for _ in range(self.number_of_partitions_x)]  # Adjust the number of queues based on your indexing logic

        for i in range(self.number_of_partitions_x):
            start_x = i
            start_y = i
            dim_x = self.window_size
            dim_y = self.window_size
            # Create a new sub matrix for each window
            sub_matrix = SubMatrix(i, i, start_x, start_y, dim_x, dim_y)
            queue_index = i  # The queue index is directly set to i
            self.sub_matrix_queues[queue_index].put(sub_matrix)


    def determine_queue_index(self, start_x, start_y):
        """
        Determine the queue index for a given start_x and start_y based on the processing order.
        """
        if self.processing_order == Diagonal:
            return start_x + start_y
        elif self.processing_order == Vertical:
            return start_y
        elif self.processing_order == Bulk:
            return 0
        else:
            # Implement additional processing order logic if necessary
            return 0

    def process_sub_matrix_queue(self, **kwargs):
        """
        Processing of a single sub matrix queue.
        """

        device = kwargs['device']
        sub_matrix_queue = kwargs['sub_matrix_queue']
        
        while True:
            try:
                # Get sub matrix from queue
                sub_matrix = sub_matrix_queue.get(False)

                # Extend sub matrix with necessary data
                self.extend_sub_matrix(sub_matrix)

                # Process sub matrix
                sub_matrix_processed = False
                while not sub_matrix_processed:
                    try:
                        flavour_runtimes = sub_matrix.variant_instance.process_sub_matrix(sub_matrix)
                        sub_matrix_processed = True
                    except SubMatrixNotProcessedException as error:
                        print(error)
                        break
                
                # Update global data structures with the RQA measures computed for the sub_matrix
                self.update_global_data_structures(device, sub_matrix, flavour_runtimes)

            except queue.Empty:
                # No more sub-matrices to process in the queue
                break

    def run(self):
        self.reset()  # Reset global data structures
        
        # Initialize OpenCL devices if not already done
        self.validate_opencl()
        self.validate_variants()
        
        # List to store RQAResult objects for each sub-matrix
        rqa_results = []
        
        # Process each queue of sub-matrices
        for sub_matrix_queue in self.sub_matrix_queues:
            while not sub_matrix_queue.empty():
                self.reset()  # Reset global data structures
                sub_matrix = sub_matrix_queue.get()
                self.process_sub_matrix_queue(sub_matrix=sub_matrix, device=self.opencl.devices[0])
                
                # Assuming process_sub_matrix_queue or another method updates the sub_matrix with RQA measures
                # Create an RQAResult object for the sub-matrix
                rqa_result = RQAResult(self.settings,
                                       self.matrix_runtimes,
                                       recurrence_points=sub_matrix.recurrence_points,
                                       diagonal_frequency_distribution=sub_matrix.diagonal_frequency_distribution,
                                       vertical_frequency_distribution=sub_matrix.vertical_frequency_distribution,
                                       white_vertical_frequency_distribution=sub_matrix.white_vertical_frequency_distribution)
                rqa_results.append(rqa_result)
        
        return rqa_results
# Usage example

from pyrqa.scalable_recurrence_analysis import RQA, \
    Carryover, \
    AdaptiveImplementationSelection
from pyrqa.selector import SingleSelector
from pyrqa.variants.engine import BaseEngine
from pyrqa.variants.rqa.radius.column_no_overlap_materialisation_byte_no_recycling import ColumnNoOverlapMaterialisationByteNoRecycling

from pyrqa.variants.engine import BaseEngine
from pyrqa.scalable_recurrence_analysis import RQA, Carryover, AdaptiveImplementationSelection
from pyrqa.variants.rqa.radius.engine import Engine
from pyrqa.variants.rqa.radius.baseline import Baseline
from pyrqa.result import RQAResult
import copy

class WRQAEngine(Engine): #BaseEngine, RQA, Carryover, AdaptiveImplementationSelection):
    def __init__(self, settings, window_size=1024, step_size=1, verbose=False, opencl=None, use_profiling_events_time=True, selector=SingleSelector(loop_unroll_factors=(1,)),
                 variants=(ColumnNoOverlapMaterialisationByteNoRecycling,)):
        # BaseEngine.__init__(self, settings, verbose, window_size, Diagonal, opencl, use_profiling_events_time)
        # RQA.__init__(self, settings, verbose)
        # Carryover.__init__(self, settings, verbose)
        # AdaptiveImplementationSelection.__init__(self, settings, selector, variants, None)
        # Baseline.__init__(self, settings, verbose)
        
        Engine.__init__(self, settings, window_size,
                 processing_order=Diagonal,
                 verbose=False,
                 opencl=None,
                 use_profiling_events_time=True,
                 selector=SingleSelector(loop_unroll_factors=(1,)),
                 variants=(ColumnNoOverlapMaterialisationByteNoRecycling,),
                 variants_kwargs=None)
                 
        self.window_size = window_size
        self.step_size = step_size
        self.create_windowed_sub_matrices()
        self.__initialise()

    def __initialise(self):
        """
        Initialise the compute device-specific global data structures.
        """
        self.validate_opencl()
        self.validate_variants()

        self.device_selector = {}

        self.device_vertical_frequency_distribution = {}
        self.device_white_vertical_frequency_distribution = {}
        self.device_diagonal_frequency_distribution = {}

        for device in self.opencl.devices:
            self.device_selector[device] = copy.deepcopy(self.selector)
            self.device_selector[device].setup(device,
                                               self.settings,
                                               self.opencl,
                                               self.variants,
                                               self.variants_kwargs)

            self.device_vertical_frequency_distribution[device] = self.get_empty_global_frequency_distribution()
            self.device_white_vertical_frequency_distribution[device] = self.get_empty_global_frequency_distribution()
            self.device_diagonal_frequency_distribution[device] = self.get_empty_global_frequency_distribution()

    def reset(self):
        """
        Reset the global data structures.
        """
        RQA.reset(self)
        Carryover.reset(self)

        self.__initialise()

    def get_empty_local_frequency_distribution(self):
        """
        Get empty local frequency distribution.

        :returns: Empty local frequency distribution.
        """
        return np.zeros(self.settings.max_number_of_vectors,
                        dtype=np.uint32)

    @staticmethod
    def recurrence_points_start(sub_matrix):
        """
        Start index of the sub matrix specific segment within the global recurrence point array.

        :param sub_matrix: Sub matrix.
        :return: Recurrence points start.
        """
        return sub_matrix.start_x

    @staticmethod
    def recurrence_points_end(sub_matrix):
        """
        End index of the sub matrix specific segment within the global recurrence point array.

        :param sub_matrix: Sub matrix.
        :return: Recurrence points end.
        """
        return sub_matrix.start_x + sub_matrix.dim_x

    def get_recurrence_points(self,
                              sub_matrix):
        """
        Get the sub matrix specific segment within the global recurrence points array.

        :param sub_matrix: Sub matrix.
        :returns: Sub array of the global recurrence points array.
        """
        start = self.recurrence_points_start(sub_matrix)
        end = self.recurrence_points_end(sub_matrix)

        return self.recurrence_points[start:end]

    def set_recurrence_points(self,
                              sub_matrix):
        """
        Set the sub matrix specific segment within the global recurrence points array.

        :param sub_matrix: Sub matrix.
        """
        start = self.recurrence_points_start(sub_matrix)
        end = self.recurrence_points_end(sub_matrix)

        self.recurrence_points[start:end] = sub_matrix.recurrence_points

    def update_global_data_structures(self,
                                      device,
                                      sub_matrix):
        self.set_recurrence_points(sub_matrix)

        self.set_vertical_length_carryover(sub_matrix)
        self.set_white_vertical_length_carryover(sub_matrix)
        self.set_diagonal_length_carryover(sub_matrix)

        self.device_vertical_frequency_distribution[device] += sub_matrix.vertical_frequency_distribution
        self.device_white_vertical_frequency_distribution[device] += sub_matrix.white_vertical_frequency_distribution
        self.device_diagonal_frequency_distribution[device] += sub_matrix.diagonal_frequency_distribution

    def create_windowed_sub_matrices(self):
        """
        Create sub matrices based on the windowing approach.
        """
        self.number_of_partitions_x = (len(self.settings.time_series_y.data) - self.window_size) // self.step_size + 1
        self.number_of_partitions_y = self.number_of_partitions_x  # Assuming square sub-matrices for simplicity
        self.sub_matrix_queue = [queue.Queue() for _ in range(self.number_of_partitions_x)]  # Adjust the number of queues based on your indexing logic

        for i in range(self.number_of_partitions_x):
            start_x = i
            start_y = i
            dim_x = self.window_size
            dim_y = self.window_size
            # Create a new sub matrix for each window
            sub_matrix = SubMatrix(i, i, start_x, start_y, dim_x, dim_y)
            queue_index = i  # The queue index is directly set to i
            self.sub_matrix_queue[queue_index].put(sub_matrix)

    def process_sub_matrix_queue(self, **kwargs):
        """
        Processing of a single sub matrix queue.
        """

        device = kwargs['device']
        sub_matrix_queue = kwargs['sub_matrix_queue']
        
        # Process sub matrix
        sub_matrix_processed = False
        flavour = None
        flavour_runtimes = None
                
        while not sub_matrix_processed:
            flavour = self.get_selector(device).get_flavour()
            try:
                # Get sub matrix from queue
                sub_matrix = sub_matrix_queue.get(False)

                sub_matrix.recurrence_points = self.get_recurrence_points(sub_matrix)
                
                # Extend sub matrix with necessary data
                # self.extend_sub_matrix(sub_matrix)

                # Process sub matrix
                sub_matrix_processed = False
                while not sub_matrix_processed:
                    try:
                        flavour_runtimes = flavour.variant_instance.process_sub_matrix(sub_matrix)
                        if not self.use_profiling_events_time:
                            flavour_runtimes = FlavourRuntimesMonolithic(execute_computations=end_time - start_time)

                        sub_matrix_processed = True
                        self.get_selector(device).increment_sub_matrix_count()
                        
                        
  
                    except SubMatrixNotProcessedException as error:
                        if self.get_selector(device).__class__ == SingleSelector:
                            self.print_out(error)
                            flavour_runtimes = FlavourRuntimesMonolithic()
                            break
                        break
                
                # Update flavour runtimes
                flavour.update_runtimes(sub_matrix,
                                        flavour_runtimes)

                # Update matrix runtimes
                self.matrix_runtimes.update_sub_matrix(sub_matrix,
                                                       flavour_runtimes)
                
                # Update global data structures with the RQA measures computed for the sub_matrix
                self.update_global_data_structures(device, sub_matrix, flavour_runtimes)

            except queue.Empty:
                # No more sub-matrices to process in the queue
                break
            
                        
    def run(self):
        self.reset()  # Reset global data structures
        
        # self.run_device_selection()
        
        # Initialize OpenCL devices if not already done
        self.validate_opencl()
        self.validate_variants()
        
        # List to store RQAResult objects for each sub-matrix
        rqa_results = []
        
        # Process each queue of sub-matrices
        # while not self.sub_matrix_queue.empty():
        for sub_matrix_queue in self.sub_matrix_queue:
            while not sub_matrix_queue.empty():
                # sub_matrix = sub_matrix_queue.get()
                self.process_sub_matrix_queue(sub_matrix_queue=sub_matrix_queue, device=self.opencl.devices[0])
                
                # Assuming process_sub_matrix_queue or another method updates the sub_matrix with RQA measures
                # Create an RQAResult object for the sub-matrix
                rqa_result = RQAResult(self.settings,
                                       self.matrix_runtimes,
                                       recurrence_points=self.recurrence_points,
                                       diagonal_frequency_distribution=self.diagonal_frequency_distribution,
                                       vertical_frequency_distribution=self.vertical_frequency_distribution,
                                       white_vertical_frequency_distribution=self.white_vertical_frequency_distribution)
                rqa_results.append(rqa_result)
        
        return rqa_results
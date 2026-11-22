# This software is to simulate quantized ABBA

import os
import itertools
import warnings
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool as Pool
from sklearn.cluster import KMeans
from .fkmns import sampledKMeansInter
from .quant import QUANT
from joblib import parallel_backend


import multiprocessing
import subprocess

def get_macos_thread_affinity():
    """Attempt to get thread affinity on macOS using Mach thread_policy_get."""
    import ctypes
    from ctypes import c_uint32, c_int, c_size_t
    try:
        libc = ctypes.CDLL('libc.dylib')
        THREAD_AFFINITY_POLICY = 4
        THREAD_AFFINITY_POLICY_COUNT = 1

        class ThreadAffinityPolicyData(ctypes.Structure):
            _fields_ = [("affinity_tag", c_uint32)]

        mach_thread_self = libc.mach_thread_self
        mach_thread_self.restype = c_uint32

        policy_data = ThreadAffinityPolicyData()
        policy_count = c_uint32(THREAD_AFFINITY_POLICY_COUNT)
        get_default = c_int(0)

        ret = libc.thread_policy_get(
            mach_thread_self(),
            c_uint32(THREAD_AFFINITY_POLICY),
            ctypes.byref(policy_data),
            ctypes.byref(policy_count),
            ctypes.byref(get_default)
        )

        if ret == 0 and policy_data.affinity_tag != 0:
            # print(f"macOS thread_policy_get succeeded: Affinity tag {policy_data.affinity_tag}")
            return [policy_data.affinity_tag]
        else:
            print(f"macOS thread_policy_get failed: Error code {ret}, Affinity tag: {policy_data.affinity_tag}")
            return None
        
    except Exception as e:
        print(f"Exception in macOS thread_policy_get: {str(e)}")
        return None

def get_macos_cpu_count():
    """Get the number of available CPUs on macOS using multiprocessing."""
    try:
        cpu_count = multiprocessing.cpu_count()
        # print(f"Retrieved CPU count via multiprocessing: {cpu_count}")
        return list(range(cpu_count))
    except Exception as e:
        print(f"Exception in multiprocessing.cpu_count: {str(e)}")
        return None

def get_macos_cpu_count_sysctl():
    """Get the number of available CPUs on macOS using sysctl."""
    try:
        result = subprocess.run(['sysctl', '-n', 'hw.ncpu'], capture_output=True, text=True, check=True)
        cpu_count = int(result.stdout.strip())
        # print(f"Retrieved CPU count via sysctl: {cpu_count}")
        return list(range(cpu_count))
    except Exception as e:
        print(f"Exception in sysctl hw.ncpu: {str(e)}")
        return None

def get_cpu_affinity():
    """Get CPU affinity for the current process, with enhanced macOS support."""
    import platform
    system = platform.system()
    try:
        import psutil
    except ImportError:
        psutil = None
        
    try:
        # Linux: Try os.sched_getaffinity
        if hasattr(os, 'sched_getaffinity') and system == 'Linux':
            affinity = os.sched_getaffinity(0)
            # print(f"Retrieved Linux affinity: {affinity}")
            return affinity

        # Cross-platform: Try psutil.cpu_affinity
        if psutil is not None:
            process = psutil.Process()
            if hasattr(process, 'cpu_affinity'):
                try:
                    affinity = process.cpu_affinity()
                    # print(f"Retrieved psutil affinity: {affinity}")
                    return affinity
                except Exception as e:
                    print(f"psutil.cpu_affinity failed: {str(e)}")
            else:
                print("psutil.cpu_affinity not available in this version")
               

        # macOS: Try thread affinity and fallbacks
        if system == 'Darwin':
            # print("Attempting macOS thread affinity retrieval...")
            affinity = get_macos_thread_affinity()
            if affinity:
                return affinity
            # print("Falling back to CPU count as macOS affinity proxy...")
            affinity = get_macos_cpu_count()
            if affinity:
                # print("Note: This is the list of available CPUs, not true affinity")
                return affinity
            # print("Falling back to sysctl CPU count...")
            affinity = get_macos_cpu_count_sysctl()
            if affinity:
                # print("Note: This is the list of available CPUs, not true affinity")
                return affinity
            # print("macOS does not provide direct CPU affinity information")

        # Other platforms
        else:
            print(f"CPU affinity not supported on {system}")


    except Exception as e:
        print(f"Unexpected error in get_cpu_affinity: {str(e)}")
        
    return None



try:
    from .compfp import compress
    from .aggfp import aggregate
    from .inversefp import inv_transform, inv_digitize, quantize, inv_compress
except ImportError:
    from .comp import compress
    from .agg import aggregate
    from .inverse import inv_transform, inv_digitize, quantize, inv_compress


def compute_storage(centers, strings, bits_for_len, bits_for_inc, bits_for_ts=32):
    """Compute storage need for representation"""
    size_centers = centers.shape[0]*bits_for_len + centers.shape[0]*bits_for_inc
    size_strings = 8 * len(strings)
    return size_centers + size_strings + bits_for_ts


def symbolsAssign(clusters, alphabet_set=0):
    """
    Automatically assign symbols to different groups, start with '!'

    Parameters
    ----------
    clusters - list or pd.Series or array
        The list of labels.

    alphabet_set - int or list
        The list of alphabet letter.

    ----------
    Return:

    string (list of string), alphabets(numpy.ndarray): for the
    corresponding symbolic sequence and for mapping from symbols to labels or
    labels to symbols, repectively.

    """

    if alphabet_set == 0:
        alphabets = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e',
                     'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j',
                     'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o',
                     'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't',
                     'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z']

    elif alphabet_set == 1:
        alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                     'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                     'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                     'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                     'w', 'x', 'y', 'z']

    elif isinstance(alphabet_set, list) and len(alphabet_set):
        alphabets = list(dict.fromkeys(alphabet_set))

    else:
        alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                     'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                     'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W', 'X', 'Y', 'Z']

    clusters = np.asarray(clusters, dtype=int)
    count = len(np.unique(clusters))
    if count > len(alphabets):
        if isinstance(alphabet_set, list):
            used = set(alphabets)
            for index in itertools.count():
                if len(alphabets) == count:
                    break
                symbol = f"{index}ABBA"
                if symbol not in used:
                    alphabets.append(symbol)
        else:
            alphabets = [
                chr(codepoint) if not chr(codepoint).isspace() else f"{codepoint}ABBA"
                for codepoint in range(33, count + 33)
            ]
    alphabets = alphabets[:count]

    alphabets = np.asarray(alphabets)
    string = alphabets[clusters]
    return string.tolist(), alphabets


        
        
@dataclass
class Model:
    """
    save ABBA model - parameters
    """
    centers: np.ndarray # store aggregation centers
    
    """ dictionary """
    alphabets: np.ndarray # labels -> symbols,  symbols -> labels

    

def general_compress(pabba, data, adjust=True, n_jobs=-1):
    if len(data.shape) > 3:
        raise ValueError("Please transform the shape of data into 1D, 2D or 3D.")
    elif len(data.shape) == 3:
        pabba.d_shape = data.shape
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    else:
        pabba.d_shape = data.shape
        
    if adjust:
        _mean = data.mean(axis=0)
        _std = data.std(axis=0)
        if np.any(_std == 0):
            _std[_std == 0] = 1
        data = (data - _mean) / _std
        strings = pabba.fit_transform(data, n_jobs=n_jobs)
        pabba.d_norm = (_mean, _std)
    else:
        pabba.d_norm = None
        strings = pabba.fit_transform(data, n_jobs=n_jobs)

    return strings



def general_decompress(pabba, strings, int_type=True, n_jobs=-1):
    reconstruction = np.array(pabba.inverse_transform(strings, n_jobs=n_jobs))
    if pabba.d_norm is not None:
        try:
            reconstruction = reconstruction*pabba.d_norm[1] + pabba.d_norm[0]
        except ValueError:
            raise ValueError("The number of symbol is not enough to reconstruct this data, please use more symbols.")
            
    if len(pabba.d_shape) == 3:
        if int_type:
            reconstruction = reconstruction.round().reshape(pabba.d_shape).astype(np.uint8)
        else:
            reconstruction = reconstruction.reshape(pabba.d_shape)
            
    return reconstruction



class XABBA(object):
    """
    Simulate XABBA in parallel
    
    Parameters
    ----------        
    tol - double, default=0.5
        Tolerance for compression.
        
    k - int, default=1
        The number of clusters (distinct symbols) specified for ABBA. 
        
    r - float, default=0.5
        The rate of data sampling to perform k-means.  
    
    alpha - double, default=0.5 
        Tolerance for digitization.
    
    init - str, default='agg'
        The clustering algorithm in digitization. optional: 'f-kmeans', 'kmeans'.
    
    sorting - str, default="norm".
        Apply sorting data before aggregation (inside digitization). Alternative option: "pca".
    
    max_len - int
        The max length of series contained in each compression pieces.
    
    max_iter - int, default=2
        The max iteration for fast k-means algorithm.

    batch_size - int, default=1024
        Size of the mini batches for mini-batch kmeans in digitization. For faster compuations, 
        you can set the batch_size greater than 256 * number of cores to enable parallelism on all cores.

    verbose - int or boolean, default=1
        Enable verbose output.
    
    partition_rate - float or int, default=None
        This parameter is to get the number of partitions of time series. 
        when this parameter is not None, the partitions will be 
        n_jobs*int(np.round(np.exp(1/self.partition_rate), 0))
    
    partition - int:
        The number of subsequences for time series to be partitioned.
         
    scl - int or float, default=1
        Scale the length of compression pieces. The larger the value is, the more important of the length information is.
        Therefore, it can solve some problem resulted from peak shift.
        
        
    Attributes
    ----------
    params: dict
        Parameters of trained model.
    
    string_ - str or list
        Contains the ABBA representation.
    """
    
    def __init__(self, tol=0.2, init='agg', k=2, r=0.5, alpha=0.1, 
                        sorting="norm", scl=1, max_iter=2,
                        bits_for_len=8, bits_for_inc=16,
                        partition_rate=None, partition=None,
                        max_len=np.inf, verbose=1, random_state=42,
                        fillna='ffill'):
        
        self.tol = tol
        self.alpha = alpha
        self.k = k
        self.scl = scl
        self.init = init
        self.max_len = max_len
        self.max_iter = max_iter
        self.sorting = sorting
        self.verbose = verbose
        self.r = r
        self.partition = partition
        self.partition_rate = partition_rate
        self.temp_symbols = None
        self.symbols_ = None
        self.d_norm = None
        self.d_shape = None
        self.eta = None # The coefficient applied to auto digitization. Needless to specify.
        self.fillna = fillna
        self.random_state = random_state
        self.bits_for_len = bits_for_len
        self.bits_for_inc = bits_for_inc
        self.recap_shape = None
        
        
    def fit_transform(self, series, n_jobs=-1, alphabet_set=0, return_start_set=False):
        """
        Fitted the numerical series and transform them into symbolic representation.
        
        Parameters
        ----------        
        series - numpy.ndarray, 2-dimension or 1-dimension
            Univariate or multivariate time series
        
        n_jobs - int, default=-1
            The mumber of processors used for parallelism. When n_jobs < 0, use all of available 
            processors the machine allows. Note: For the univariate time series, if n_jobs = 1, 
            PABBA will degenerate to fABBA, but the result may be diferent since PABBA use 
            aggregated groups starting points for reconstruction instead of aggregated groups
            centers.

        alphabet_set - int or list, default=0
            The list of alphabet letter. Here provide two different kinds of alphabet letters, namely 0 and 1.
        """
        
        self.fit(series, n_jobs=n_jobs, alphabet_set=alphabet_set)
        if return_start_set:
            return self.string_, self.start_set
        else:
            return self.string_
        
                
    def fit(self, series, n_jobs=-1, alphabet_set=0):
        """
        Fitted the numerical series.
        
        Parameters
        ----------        
        series - numpy.ndarray, 2-dimension or 1-dimension
            Univariate or multivariate time series
        
        n_jobs - int, default=-1
            The mumber of processors used for parallelism. When n_jobs < 0, use all of available 
            processors the machine allows. Note: For the univariate time series, if n_jobs = 1, 
            PABBA will degenerate to fABBA, but the result may be diferent since PABBA use 
            aggregated groups starting points for reconstruction instead of aggregated groups
            centers.

        alphabet_set - int or list, default=0
            The list of alphabet letter. Here provide two different kinds of alphabet letters, namely 0 and 1.
        """
        
        self.pieces = self.parallel_compress(series, n_jobs=n_jobs)
        self.string_ = self.digitize(series, self.pieces, alphabet_set, n_jobs)    
        return self
        
    
    
    def parallel_compress(self, series, n_jobs=-1):
        """
        Compress the numerical series in a parallel manner.
        
        Parameters
        ----------        
        series - numpy.ndarray, 2-dimension or 1-dimension
            Univariate or multivariate time series
        
        n_jobs - int, default=-1
            The mumber of processors used for parallelism. When n_jobs < 0, use all of available 
            processors the machine allows. Note: For the univariate time series, if n_jobs = 1, 
            PABBA will degenerate to fABBA, but the result may be diferent since PABBA use 
            aggregated groups starting points for reconstruction instead of aggregated groups
            centers.
        """
        

        rows, n_jobs = self._prepare_series(series, n_jobs, fitting=True)
        self.start_set = [row[0] for row in rows]
        inputs = [(row, self.tol, self.max_len) for row in rows]
        if n_jobs == 1:
            return [compress(*args) for args in inputs]
        with Pool(n_jobs) as pool:
            return pool.starmap(compress, inputs)

    def _prepare_series(self, series, n_jobs, fitting=False, llm_split="Pre"):
        values = np.asarray(series, dtype=np.float64)
        if values.ndim == 0 or not values.size:
            raise ValueError("Expected a nonempty time series or batch of time series.")
        if llm_split not in {"Pre", "Post"}:
            raise ValueError("llm_split must be 'Pre' or 'Post'.")

        self.return_series_univariate = values.ndim == 1
        if fitting:
            self.recap_shape = values.shape if values.ndim > 2 else None

        if self.return_series_univariate:
            max_parts = max(1, len(values) // 2)
            n_jobs = self.n_jobs_init(n_jobs, _max=max_parts)
            partition = n_jobs
            if self.partition is not None:
                partition = self.partition
            elif self.partition_rate is not None:
                # Cap the exponent before it can overflow for very small rates.
                max_multiplier = max_parts / n_jobs
                exponent = min(1 / self.partition_rate, np.log(max_multiplier))
                partition = int(round(np.exp(exponent))) * n_jobs
            rows = np.array_split(values, min(partition, max_parts))
        else:
            if (values.ndim > 2 and llm_split == "Pre" and self.recap_shape is not None
                    and values.shape[1:] != self.recap_shape[1:]):
                raise ValueError("Expected the same time and feature dimensions as in fit.")
            rows = values.reshape(len(values), -1)
            n_jobs = self.n_jobs_init(n_jobs, _max=len(rows))

        rows = [fillna(row.copy(), self.fillna) for row in rows]
        return rows, min(n_jobs, len(rows))

    def digitize(self, series, pieces, alphabet_set=0, n_jobs=-1):
        """ Digitization 
        
        Parameters
        ---------- 
        pieces - numpy.ndarray
            The compressed pieces of numpy.ndarray with shape (n_samples, n_features) after compression

        len_ts - int
            The length of time series.
            
        num_pieces - int
            The number of pieces.
            
        init - str
            Use aggregation, fast-kmeans or kmeans for digitization to get symbols.
            
        alphabet_set - int or list, default=0
            The list of alphabet letter. Here provide two different kinds of alphabet letters, namely 0 and 1.
        """
        
        self.eta = 0.01 if np.ndim(series) == 1 else 0.000002
        length_scale = self.scl or 1

        num_pieces = list()

        for i in range(len(pieces)):
            num_pieces.append(len(pieces[i]))
        
        pieces = np.vstack(pieces)[:,:2]
        self._std = np.std(pieces, axis=0)
        if self._std[0] == 0: # prevent zero-division
            self._std[0] = 1
        if self._std[1] == 0:
            self._std[1] = 1

        if self.scl == 0:
            len_pieces = pieces[:,0]

        pieces = pieces * np.array([self.scl, 1]) / self._std
        max_k = np.unique(pieces[:,:2],axis=0).shape[0]

        if self.init == 'agg':
            labels, splist = aggregate(pieces, self.sorting, self.alpha)
            splist = np.array(splist)
            centers = splist[:,3:5] * self._std / np.array([length_scale, 1])
            self.k = centers.shape[0]
            
        elif self.init == 'f-kmeans':
            if self.k > max_k:
                self.k = max_k
                warnings.warn("k is larger than the unique pieces size, so k reduces to unique pieces size.")
            

            with parallel_backend('threading', n_jobs=n_jobs):
                kmeans = sampledKMeansInter(n_clusters=self.k, 
                                            r=self.r, 
                                            init='k-means++', 
                                            max_iter=self.max_iter, 
                                            random_state=self.random_state)
                kmeans.sampled_fit(pieces)
                
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_ * self._std / np.array([length_scale, 1])

        else: # default => 'kmeans'
            if self.k > max_k:
                self.k = max_k
                warnings.warn("k is larger than the unique pieces size, so k reduces to unique pieces size.")
            
            with parallel_backend('threading', n_jobs=n_jobs):
                kmeans = KMeans(n_clusters=self.k, random_state=0, n_init=1).fit(pieces)
                
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_ * self._std / np.array([length_scale, 1])
            splist = None
        
        if self.scl == 0:
            centers[:, 0] = one_D_centers(len_pieces, labels, self.k)

        self.quantizer_len = QUANT(bits=self.bits_for_len)
        self.quantizer_inc = QUANT(bits=self.bits_for_inc)
        centers[:, 0] = self.quantizer_len(centers[:, 0])
        centers[:, 1] = self.quantizer_inc(centers[:, 1])

        string, alphabets = symbolsAssign(labels, alphabet_set)
        self.parameters = Model(centers, alphabets)
        
        self.num_grp = self.parameters.centers.shape[0]
        if self.verbose:
            print("Generate {} symbols".format(self.num_grp))
        
        string_sequences = self.string_separation(string, num_pieces)
        return string_sequences
        
        
        
    def transform(self, series, n_jobs=-1, llm_split='Pre'):
        """
        Transform multiple series (numerical sequences) to symbolic sequences.
        
        Parameters
        ----------        
        series: numpy.ndarray, 2-dimension or 1-dimension
            Univariate or multivariate time series
        
        n_jobs: int, default=-1
            The mumber of processors used for parallelism. When n_jobs < 0, use all of processors 
            the machine allows.  Note: if n_jobs = 1, PABBA will degenerate to fABBA for transfomation.

        """
        
        rows, n_jobs = self._prepare_series(series, n_jobs, llm_split=llm_split)
        start_set = [row[0] for row in rows]
        if n_jobs == 1:
            string_sequences = [self.transform_single_series(row) for row in rows]
        else:
            with Pool(n_jobs) as pool:
                string_sequences = pool.map(self.transform_single_series, rows)
        return string_sequences, start_set

    def transform_single_series(self, series):
        """
        Transform a single series to symbols.
        
        Parameters
        ----------        
        series: numpy.ndarray, 1-dimension
            Univariate time series
        """
        
        pieces = compress(series, self.tol, self.max_len)
        pieces = pieces[:,:2] 

        symbols_series = list()
        
        for piece_i in pieces:
            symbols_series.append(self.piece_to_symbol(piece_i))
        
        return symbols_series

        
        
    def recast_shape(self, reconstruct_list, recap_shape=None):
        """Pad or truncate reconstructed rows to the fitted or supplied batch shape."""
        shape = self.recap_shape if recap_shape is None else tuple(recap_shape)
        if shape is None or len(shape) < 2:
            raise ValueError("A batch shape is required to reshape reconstructed series.")

        row_length = int(np.prod(shape[1:]))
        if any(len(row) < row_length for row in reconstruct_list):
            warnings.warn("The reconstructed shape has been expanded.", ShapeWarning)
        if any(len(row) > row_length for row in reconstruct_list):
            warnings.warn("The reconstructed shape has been squeezed.", ShapeWarning)

        rows = np.empty((len(reconstruct_list), row_length))
        for index, row in enumerate(reconstruct_list):
            width = min(len(row), row_length)
            rows[index, :width] = row[:width]
            if width < row_length:
                rows[index, width:] = np.mean(row) if width else 0
        return rows.reshape(len(rows), *shape[1:])

    def inverse_transform(self, string_sequences, start_set=None, n_jobs=1):
        """
        Reconstruct the symbolic sequences to numerical sequences.
        
        Parameters
        ----------        
        string_sequences: list
            Univariate or multivariate symbolic time series
        
        start_set: list
            starting value for each symbolic time series reconstruction.
            
        hstack: boolean, default=False
            Determine if concate multiple reconstructed time series into a single time series, 
            which will be useful in the parallelism in univariate time series reconstruction.
            
        n_jobs: int, default=-1
            The mumber of processors used for parallelism. When n_jobs < 0, use all of processors 
            the machine allows.
        """
        
        count = len(string_sequences)
        if count == 0:
            return np.empty(0) if self.return_series_univariate else []
        n_jobs = self.n_jobs_init(n_jobs, _max=count)
        
        if start_set is None:
            start_set = self.start_set
            if start_set is None:
                raise ValueError('Please input valid start_set.')
        if len(start_set) != count:
            raise ValueError('Expected one start value per symbolic sequence.')

        inverse_sequences = list()
        
        centers = np.zeros(self.parameters.centers.shape)
        centers[:, 0] = self.quantizer_len.dequant(self.parameters.centers[:, 0])
        centers[:, 1] = self.quantizer_inc.dequant(self.parameters.centers[:, 1])
        
        if n_jobs != 1 and count != 1:
            p = Pool(n_jobs)
            for i in range(count):
                inverse_sequences.append(
                    p.apply_async(inv_transform,
                                  args=(string_sequences[i],
                                        centers,
                                        self.parameters.alphabets.tolist(),
                                        start_set[i])
                                 )
                )
            p.close()
            p.join()        
            inverse_sequences = [p.get() for p in inverse_sequences]
        else:
            for i in range(count):
                inverse_sequence = inv_transform(string_sequences[i], 
                                                 centers,
                                                 self.parameters.alphabets.tolist(),
                                                 start_set[i]
                                                )
                inverse_sequences.append(inverse_sequence)

        if self.return_series_univariate:
            inverse_sequences = np.hstack(inverse_sequences)
            
        return inverse_sequences
        

    
    
    def piece_to_symbol(self, piece):
        """
        Transform a piece to symbol.
        
        Parameters
        ----------        
        piece: numpy.ndarray
            A piece from compression pieces.

        """
        
        
        centers = self.parameters.centers.copy()
        centers[:, 0] = self.quantizer_len.dequant(centers[:, 0])
        centers[:, 1] = self.quantizer_inc.dequant(centers[:, 1])
        
        offsets = (centers - piece) * np.array([self.scl, 1]) / self._std
        splabels = np.argmin(np.linalg.norm(offsets, axis=1))
        return self.parameters.alphabets[splabels]
    
    
    
    
    def string_separation(self, symbols, num_pieces):
        """
        Separate symbols into symbolic subsequence.
        """
        
        string_sequences = list()
        num_pieces_csum = np.cumsum([0] + num_pieces)
        for index in range(len(num_pieces)):
            string_sequences.append(symbols[num_pieces_csum[index]:num_pieces_csum[index+1]])
        return string_sequences



    def n_jobs_init(self, n_jobs=-1, _max=np.inf):
        """
        Initialize parameter n_jobs.
        """
        
        if n_jobs is None or (isinstance(n_jobs, int) and n_jobs < 0):
            affinity = get_cpu_affinity()
            n_jobs = len(affinity) if affinity else 1
        if not isinstance(n_jobs, int):
            raise TypeError("n_jobs must be an integer.")
        if n_jobs == 0:
            raise ValueError("n_jobs must be nonzero.")
        return min(n_jobs, _max)

    @property
    def tol(self):
        return self._tol
    
    
    
    @tol.setter
    def tol(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("Expected a float or int type.")
        if value <= 0:
            raise ValueError(
                "Please feed an correct value for tolerance.")
        if value > 1:
            warnings.warn("Might lead to bad aggregation.", DeprecationWarning)
        self._tol = value
    
    
        
    @property
    def sorting(self):
        return self._sorting
    
    
    
    @sorting.setter
    def sorting(self, value):
        if not isinstance(value, str):
            raise TypeError("Expected a string type")
        if value not in ["lexi", "2-norm", "1-norm", "norm", "pca"]:
            raise ValueError(
                "Please refer to an correct sorting way, namely 'lexi', '2-norm' and '1-norm'.")
        self._sorting = value
        


    @property
    def alpha(self):
        return self._alpha
    
    
    
    @alpha.setter
    def alpha(self, value):
        if value != None:
            if not isinstance(value, float) and not isinstance(value,int):
                raise TypeError("Expected a float or int type.")
        
            if value <= 0:
                raise ValueError(
                    "Please feed an correct value for alpha.")

        self._alpha = value



    @property
    def max_len(self):
        return self._max_len



    @max_len.setter
    def max_len(self, value):
        if value != np.inf:
            if not isinstance(value, float) and not isinstance(value,int):
                raise TypeError("Expected a float or int type.")
        
        if not value >= 1:
            raise ValueError("max_len must be at least 1.")

        self._max_len = value
        
        

    @property
    def partition(self):
        return self._partition



    @partition.setter
    def partition(self, value):
        if value != None:
            if not isinstance(value, int):
                raise TypeError("Expected int type.")
        
            if value <= 0:
                raise ValueError(
                    "Please feed an correct value for partition.")

        self._partition = value
        
        
        
    @property
    def partition_rate(self):
        return self._partition_rate



    @partition_rate.setter
    def partition_rate(self, value):
        if value != None:
            if not isinstance(value, float) and not isinstance(value,int):
                raise TypeError("Expected float or int type.")
        
            if not 0 < value <= 1:
                raise ValueError(
                    "Please feed an correct value for partition_rate.")

        self._partition_rate = value
        
        

    @property
    def scl(self):
        return self._scl



    @scl.setter
    def scl(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("Expected float or int type.")

        if value < 0:
            raise ValueError(
                "Please feed an correct value for scl.")

        self._scl = value
        
        
    @property
    def eta(self):
        return self._eta



    @eta.setter
    def eta(self, value):
        if value is not None:
            if not isinstance(value, float) and not isinstance(value,int):
                raise TypeError("Expected float or int type.")

        if value == 0:
            raise ValueError(
                "Please feed an correct value for eta.")

        self._eta = value
        
        
        
    @property
    def k(self):
        return self._k
        
        
        
    @k.setter
    def k(self, value):
        if not isinstance(value,int):
            raise TypeError("Expected int type.")

        if value == 0:
            raise ValueError(
                "Please feed an correct value for k.")

        self._k = value
        
        




class fastXABBA(object): # simulate single time series
    def __init__(self, tol=0.2, k=2, r=1, scl=1, init='agg', 
                 sorting="norm", alpha=0.5,
                 bits_for_len=8, bits_for_inc=16, random_state=2022, n_jobs=1, 
                 alphabet_set=0, max_len=np.inf, max_iter=2, verbose=True):
        
        self.tol = tol
        self.k = k
        self.r = r
        self.scl = scl
        self.max_len = max_len
        self.sorting, self.alpha = sorting, alpha
        self.bits_for_len = bits_for_len
        self.bits_for_inc = bits_for_inc
        self.random_state = random_state
        self.verbose = verbose
        self.alphabet_set = alphabet_set
        self.init = init
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def transform(self, series):
        series = np.asarray(series, dtype=np.float64)
        pieces = self._compress(series)
        return self._digitize(pieces)


    def predict(self, series):
        if self.init == 'kmeans':
            if self.kmeans is not None:
                series = np.asarray(series, dtype=np.float64)
                pieces = self._compress(series)
                pieces = pieces[:,:2]
                pieces = pieces * np.array([self.scl, 1]) / self._std

                with parallel_backend('threading', n_jobs=self.n_jobs):
                    labels = self.kmeans.predict(pieces)

                return self.parameters.alphabets[labels]
        
        elif self.init == 'agg':
            raise ValueError("Underdeveloped.")
        else:
            raise ValueError("Please train the model first.")


    def _compress(self, series):
        return compress(series, tol=self.tol, max_len=self.max_len)


    def _digitize(self, pieces):
        pieces = pieces[:,:2]
        max_k = np.unique(pieces, axis=0).shape[0]
        if self.k > max_k:
            self.k = max_k
            warnings.warn("k is larger than the unique pieces size, so k reduces to unique pieces size.")
            
        self._std = np.std(pieces, axis=0)
        if self._std[0] == 0: # prevent zero-division
            self._std[0] = 1
        if self._std[1] == 0:
            self._std[1] = 1

        if self.scl == 0:
            len_pieces = pieces[:, 0]
        
        pieces =  pieces * np.array([self.scl, 1]) / self._std
        
        if self.init == 'kmeans':
            with parallel_backend('threading', n_jobs=self.n_jobs):
                self.kmeans = sampledKMeansInter(n_clusters=self.k, 
                                                r=self.r, 
                                                init='k-means++', 
                                                max_iter=self.max_iter, 
                                                random_state=self.random_state)

                labels = self.kmeans.sampled_fit_predict(pieces[:,:2])

            if self.scl != 0:
                centers = self.kmeans.cluster_centers_ * self._std / np.array([self.scl, 1])
            else:
                centers = self.kmeans.cluster_centers_ * self._std
                centers[:, 0] = one_D_centers(len_pieces, labels, self.k)
        
        else:
            labels, splist = aggregate(pieces, self.sorting, self.alpha)
            splist = np.array(splist)
            
            if self.scl != 0:
                centers = splist[:,3:5] * self._std / np.array([self.scl, 1])
            else:
                centers = splist[:,3:5] * self._std
                centers[:, 0] = one_D_centers(len_pieces, labels, self.k)
            
            self.k = centers.shape[0]
            
        self.quantizer_len = QUANT(bits=self.bits_for_len)
        self.quantizer_inc = QUANT(bits=self.bits_for_inc)
        centers[:, 0] = self.quantizer_len(centers[:, 0])
        centers[:, 1] = self.quantizer_inc(centers[:, 1])
        
        self.string_, alphabets = symbolsAssign(labels, self.alphabet_set)
        self.parameters = Model(centers, alphabets)
        
        self.num_grp = self.parameters.centers.shape[0]
        
        if self.verbose:
            print("Generate {} symbols".format(self.num_grp))
        
        return self.string_


    def inverse_transform(self, strings, start):
        centers = np.zeros(self.parameters.centers.shape)
        centers[:, 0] = self.quantizer_len.dequant(self.parameters.centers[:, 0])
        centers[:, 1] = self.quantizer_inc.dequant(self.parameters.centers[:, 1])
        
        return inv_transform(strings, centers, self.parameters.alphabets.tolist(), start)


class fastXABBA_len(object): # simulate single time series
    def __init__(self, tol=0.2, k=2, r=1, scl=1, init='agg', 
                 sorting="norm", alpha=0.5,
                 bits_for_len=8, bits_for_inc=16, random_state=2022, n_jobs=1, 
                 alphabet_set=0, max_len=np.inf, max_iter=2, verbose=True):
        
        self.tol = tol
        self.k = k
        self.r = r
        self.scl = scl
        self.max_len = max_len
        self.sorting, self.alpha = sorting, alpha
        self.bits_for_len = bits_for_len
        self.bits_for_inc = bits_for_inc
        self.random_state = random_state
        self.verbose = verbose
        self.alphabet_set = alphabet_set
        self.init = init
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def transform(self, series):
        series = np.asarray(series, dtype=np.float64)
        pieces = self._compress(series)
        return self._digitize(pieces)


    def predict(self, series):
        if self.init == 'kmeans':
            if self.kmeans is not None:
                series = np.asarray(series, dtype=np.float64)
                pieces = self._compress(series)
                pieces = pieces[:,:2]
                pieces = pieces * np.array([self.scl, 1]) / self._std

                with parallel_backend('threading', n_jobs=self.n_jobs):
                    labels = self.kmeans.predict(pieces)

                return self.parameters.alphabets[labels]
        
        elif self.init == 'agg':
            raise ValueError("Underdeveloped.")
        else:
            raise ValueError("Please train the model first.")


    def _compress(self, series):
        return compress(series, tol=self.tol, max_len=self.max_len)


    def _digitize(self, pieces):
        pieces = pieces[:,:2]
        max_k = np.unique(pieces, axis=0).shape[0]
        if self.k > max_k:
            self.k = max_k
            warnings.warn("k is larger than the unique pieces size, so k reduces to unique pieces size.")
            
        self._std = np.std(pieces, axis=0)
        if self._std[0] == 0: # prevent zero-division
            self._std[0] = 1
        if self._std[1] == 0:
            self._std[1] = 1
            
        if self.scl == 0:
            len_pieces = pieces[:, 0]
        
        pieces =  pieces * np.array([self.scl, 1]) / self._std
        
        if self.init == 'kmeans':
            with parallel_backend('threading', n_jobs=self.n_jobs):
                self.kmeans = sampledKMeansInter(n_clusters=self.k, 
                                                r=self.r, 
                                                init='k-means++', 
                                                max_iter=self.max_iter, 
                                                random_state=self.random_state)

                labels = self.kmeans.sampled_fit_predict(pieces[:,:2])

            if self.scl != 0:
                centers = self.kmeans.cluster_centers_ * self._std / np.array([self.scl, 1])
            else:
                centers = self.kmeans.cluster_centers_ * self._std
                centers[:, 0] = one_D_centers(len_pieces, labels, self.k)
        
        else:
            labels, splist = aggregate(pieces, self.sorting, self.alpha)
            splist = np.array(splist)
            
            if self.scl != 0:
                centers = splist[:,3:5] * self._std / np.array([self.scl, 1])
            else:
                centers = splist[:,3:5] * self._std
                centers[:, 0] = one_D_centers(len_pieces, labels, self.k)
            
            self.k = centers.shape[0]
            
        self.quantizer_len = QUANT(bits=self.bits_for_len)
        centers[:, 0] = self.quantizer_len(centers[:, 0])
        
        self.string_, alphabets = symbolsAssign(labels, self.alphabet_set)
        self.parameters = Model(centers, alphabets)
        
        self.num_grp = self.parameters.centers.shape[0]
        
        if self.verbose:
            print("Generate {} symbols".format(self.num_grp))
        
        return self.string_


    def inverse_transform(self, strings, start):
        centers = np.zeros(self.parameters.centers.shape)
        centers[:, 0] = self.quantizer_len.dequant(self.parameters.centers[:, 0])
        centers[:, 1] = np.float32(self.parameters.centers[:, 1])
        return inv_transform(strings, centers, self.parameters.alphabets.tolist(), start)

    

class fastXABBA_inc(object): # simulate single time series
    def __init__(self, tol=0.2, k=2, r=1, scl=1, init='agg', 
                 sorting="norm", alpha=0.5,
                 bits_for_len=8, bits_for_inc=16, random_state=2022, n_jobs=1, 
                 alphabet_set=0, max_len=np.inf, max_iter=2, verbose=True):
        
        self.tol = tol
        self.k = k
        self.r = r
        self.scl = scl
        self.max_len = max_len
        self.sorting, self.alpha = sorting, alpha
        self.bits_for_len = bits_for_len
        self.bits_for_inc = bits_for_inc
        self.random_state = random_state
        self.verbose = verbose
        self.alphabet_set = alphabet_set
        self.init = init
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def transform(self, series):
        series = np.asarray(series, dtype=np.float64)
        pieces = self._compress(series)
        return self._digitize(pieces)


    def predict(self, series):
        if self.init == 'kmeans':
            if self.kmeans is not None:
                series = np.asarray(series, dtype=np.float64)
                pieces = self._compress(series)
                pieces = pieces[:,:2]
                pieces = pieces * np.array([self.scl, 1]) / self._std

                with parallel_backend('threading', n_jobs=self.n_jobs):
                    labels = self.kmeans.predict(pieces)

                return self.parameters.alphabets[labels]
        
        elif self.init == 'agg':
            raise ValueError("Underdeveloped.")
        else:
            raise ValueError("Please train the model first.")


    def _compress(self, series):
        return compress(series, tol=self.tol, max_len=self.max_len)


    def _digitize(self, pieces):
        pieces = pieces[:,:2]
        max_k = np.unique(pieces, axis=0).shape[0]
        if self.k > max_k:
            self.k = max_k
            warnings.warn("k is larger than the unique pieces size, so k reduces to unique pieces size.")
            
        self._std = np.std(pieces, axis=0)
        if self._std[0] == 0: # prevent zero-division
            self._std[0] = 1
        if self._std[1] == 0:
            self._std[1] = 1
            
        if self.scl == 0:
            len_pieces = pieces[:, 0]
        
        pieces =  pieces * np.array([self.scl, 1]) / self._std
        
        if self.init == 'kmeans':
            with parallel_backend('threading', n_jobs=self.n_jobs):
                self.kmeans = sampledKMeansInter(n_clusters=self.k, 
                                                r=self.r, 
                                                init='k-means++', 
                                                max_iter=self.max_iter, 
                                                random_state=self.random_state)

                labels = self.kmeans.sampled_fit_predict(pieces[:,:2])

            if self.scl != 0:
                centers = self.kmeans.cluster_centers_ * self._std / np.array([self.scl, 1])
            else:
                centers = self.kmeans.cluster_centers_ * self._std
                centers[:, 0] = one_D_centers(len_pieces, labels, self.k)
        
        else:
            labels, splist = aggregate(pieces, self.sorting, self.alpha)
            splist = np.array(splist)
            
            if self.scl != 0:
                centers = splist[:,3:5] * self._std / np.array([self.scl, 1])
            else:
                centers = splist[:,3:5] * self._std
                centers[:, 0] = one_D_centers(len_pieces, labels, self.k)
            
            self.k = centers.shape[0]
            
        self.quantizer_inc = QUANT(bits=self.bits_for_inc)
        centers[:, 1] = self.quantizer_inc(centers[:, 1])
        
        self.string_, alphabets = symbolsAssign(labels, self.alphabet_set)
        self.parameters = Model(centers, alphabets)
        
        self.num_grp = self.parameters.centers.shape[0]
        
        if self.verbose:
            print("Generate {} symbols".format(self.num_grp))
        
        return self.string_


    def inverse_transform(self, strings, start):
        centers = np.zeros(self.parameters.centers.shape)
        centers[:, 0] = np.float32(self.parameters.centers[:, 0])
        centers[:, 1] = self.quantizer_inc.dequant(self.parameters.centers[:, 1])
        
        return inv_transform(strings, centers, self.parameters.alphabets.tolist(), start)






class ABBA(XABBA):
    def __init__(self, **args):
        super().__init__(**args)

    def encode(self, ts):
        return self.fit_transform(ts)

    def decode(self, strings):
        return self.inverse_transform(strings)
    



def one_D_centers(data, labels, k): # for scl = 0
    centers = np.zeros(k)
    for clust in np.unique(labels):
        centers[clust] = np.mean(data[labels == clust])

    return centers 


# data = np.random.randn(10)
# labels = np.array([1,1,1,0,1,1,1,0,0,0])
# one_D_centers(data, labels)



def dtw(x, y, *, dist=lambda a, b: (a-b)*(a-b), return_path=False, filter_redundant=False):
    """
    Compute dynamic time warping distance between two time series x and y.
    
    Parameters
    ----------
    x - list
        First time series.
    y - list
        Second time series.
    dist - lambda function
        Lambda function defining the distance between two points of the time series.
        By default we use (x-y)^2 to correspond to literature standard for
        dtw. Note final distance d should be square rooted.
    return_path - bool
        Option to return tuple (d, path) where path is a list of tuples outlining
        the route through time series taken to compute dtw distance.
    filter_redundant - bool
        Control filtering to remove `redundant` time series due to sampling
        resolution. For example, if x = [0, 1, 2, 3, 4] and y = [0, 4]. The dynamic
        time series distance is non-zero. If filter_redundant=True then we remove
        the middle 3 time points from x where gradient is constant.
   
    Returns
    -------
    d - numpy float
        Summation of the dist(x[i], y[i]) along the optimal path to minimise overall
        distance. Standard dynamic time warping distance given by default dist and
        d**(0.5).
    
    path - list
        Path taken through time series.
    """

    x = np.array(x)
    y = np.array(y)

    if filter_redundant:
        # remove points
        if len(x) > 2:
            xdiff = np.diff(x)
            x_keep = np.abs(xdiff[1:] - xdiff[0:-1]) >= 1e-14
            x = x[np.hstack((True, x_keep, True))]
        else:
            x_keep = []

        if len(y) > 2:
            ydiff = np.diff(y)
            y_keep = np.abs(ydiff[1:] - ydiff[0:-1]) >= 1e-14
            y = y[np.hstack((True, y_keep, True))]
        else:
            y_keep = []

    len_x, len_y = len(x), len(y)
    window = [(i+1, j+1) for i in range(len_x) for j in range(len_y)]
    D = defaultdict(lambda: (float('inf'),))

    if return_path:
        if filter_redundant:
            x_ind = np.arange(1, len(x_keep)+1)
            y_ind = np.arange(1, len(y_keep)+1)
            x_ind = np.hstack((0, x_ind[x_keep], len(x_keep)+1))
            y_ind = np.hstack((0, y_ind[y_keep], len(y_keep)+1))
        else:
            x_ind = np.arange(len(x))
            y_ind = np.arange(len(y))

        D[0, 0] = (0, 0, 0)
        for i, j in window:
            dt = dist(x[i-1], y[j-1])
            D[i, j] = min((D[i-1, j][0]+dt, i-1, j), (D[i, j-1][0]+dt, i, j-1),
                          (D[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])

        path = []
        i, j = len_x, len_y
        while not (i == j == 0):
            path.append((x_ind[i-1], y_ind[j-1]))
            i, j = D[i, j][1], D[i, j][2]
        path.reverse()
        return (D[len_x, len_y][0], path)

    else:
        D[0, 0] = 0
        for i, j in window:
            dt = dist(x[i-1], y[j-1])
            D[i, j] = min(D[i-1, j]+dt, D[i, j-1]+dt, D[i-1, j-1]+dt)
        return D[len_x, len_y]
    
    
    
    
    
def fillna(series, method='ffill'):
    """Fill the NA values
    
    Parameters
    ----------   
    series - numpy.ndarray or list
        Time series of the shape (1, n_samples).

    fillna - str, default = 'zero'
        Fill NA/NaN values using the specified method.
        'zero': Fill the holes of series with value of 0.
        'mean': Fill the holes of series with mean value.
        'median': Fill the holes of series with mean value.
        'ffill': Forward last valid observation to fill gap.
            If the first element is nan, then will set it to zero.
        'bfill': Use next valid observation to fill gap. 
            If the last element is nan, then will set it to zero.        
    """

    missing = np.isnan(series)
    if not missing.any():
        return series
    if missing.all():
        series.fill(0)
    elif method == 'mean':
        series[missing] = np.mean(series[~missing])

    elif method == 'median':
        series[np.isnan(series)] = np.median(series[~np.isnan(series)])

    elif method == 'ffill':
        for i in np.where(np.isnan(series))[0]:
            if i > 0:
                series[i] = series[i-1]
            else:
                series[i] = 0

    elif method == 'bfill':
        for i in sorted(np.where(np.isnan(series))[0], reverse=True):
            if i + 1 < len(series):
                series[i] = series[i+1]
            else:
                series[i] = 0
    else:
        series[np.isnan(series)] = 0

    return series



class ShapeWarning(EncodingWarning):
    pass


def zip_longest(*iterables, fillvalue=None):
    # zip_longest('ABCD', 'xy', fillvalue='-') → Ax By C- D-

    iterators = list(map(iter, iterables))
    num_active = len(iterators)
    if not num_active:
        return

    while True:
        values = []
        for i, iterator in enumerate(iterators):
            try:
                value = next(iterator)
            except StopIteration:
                num_active -= 1
                if not num_active:
                    return
                iterators[i] = itertools.repeat(fillvalue[i])
                value = fillvalue[i]
            values.append(value)
        yield tuple(values)
